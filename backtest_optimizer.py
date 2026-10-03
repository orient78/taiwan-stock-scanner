import json
import math
import os
from datetime import datetime
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import twstock
from institutional_history import attach_gate, load_reports

TIMEZONE = "Asia/Taipei"
START_DATE = os.getenv("BACKTEST_START", "2026-01-01")
TARGET_RETURN = float(os.getenv("TARGET_RETURN", "20"))
TARGET_PASS_RATE = float(os.getenv("TARGET_PASS_RATE", "80"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "120"))
RUN_ID = int(os.getenv("GITHUB_RUN_NUMBER", "0"))
HISTORY_FILE = "backtest_results/history_best.json"

# Exit rules requested by the project:
# close >= MA10 => keep the full position, even below MA5
# close < MA10 => exit the full position
# Signals are evaluated at close and executed at next day's open.

def listed_twse_codes():
    twse = getattr(twstock, "twse", {}) or {}
    codes = []
    for raw in twse:
        code = str(raw).strip()
        if code.isdigit() and len(code) == 4 and not code.startswith("0") and code in twstock.codes:
            codes.append(code)
    codes = sorted(set(codes))
    if "2330" not in codes:
        raise RuntimeError("2330 missing from TWSE universe")
    return codes

def indicators(df):
    x = df.copy()
    for n in (5, 10, 20, 60):
        x[f"MA{n}"] = x["Close"].rolling(n).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = (100 - 100 / (1 + rs)).mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["HIGH20_PREV"] = x["High"].shift(1).rolling(20).max()
    # Keep the trading calendar intact: an invalid indicator skips a signal,
    # never changes the next session used for execution.
    return x

def select_symbols(codes, limit):
    """Cover the code range deterministically; zero requests the full universe."""
    codes = sorted(set(codes))
    if not codes or "2330" not in codes:
        raise ValueError("TWSE universe must include 2330")
    if limit < 0:
        raise ValueError("MAX_SYMBOLS must be nonnegative")
    if limit == 0 or limit >= len(codes):
        return codes
    if limit == 1:
        return ["2330"]
    # Reserve one slot for 2330, then sample across all remaining codes.
    others = [code for code in codes if code != "2330"]
    indices = np.linspace(0, len(others) - 1, limit - 1, dtype=int)
    return sorted(["2330", *(others[i] for i in indices)])

def entry_signal(row, p):
    if p.get("institutional_2d", True) and not row.get("INSTITUTIONAL_2D", False):
        return False
    price = float(row["Close"])
    if not (price > row["MA5"] and price > row["MA10"] and price > row["MA20"]):
        return False
    if p["ma_order"] and not (row["MA5"] >= row["MA10"]):
        return False
    if not (p["rsi_min"] <= row["RSI"] <= p["rsi_max"]):
        return False
    vol_ratio = row["Volume"] / row["VOL20"] if row["VOL20"] else 0
    if vol_ratio < p["vol_ratio"]:
        return False
    if p["breakout_pct"] is not None:
        threshold = row["HIGH20_PREV"] * (1 + p["breakout_pct"] / 100)
        if price < threshold:
            return False
    return True

def backtest_one(df, p, prepared=False, start_date=None, end_date=None):
    if p.get("ma10_confirm", 1) != 1:
        raise ValueError("MA10 exit must execute after one close below the MA")
    df = df if prepared else indicators(df)
    source = df
    start = pd.Timestamp(start_date or START_DATE, tz=df.index.tz)
    df = df[df.index >= start]
    if end_date is not None:
        df = df[df.index < pd.Timestamp(end_date, tz=df.index.tz)]
    if len(df) < 25:
        return None
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Trading dates must be sorted and unique")
    if not np.isfinite(df[["Open", "Close"]].to_numpy()).all() or (df[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Invalid execution/valuation prices")

    buy_cost = float(os.getenv("BUY_COST_BPS", "14.25")) / 10000
    sell_cost = float(os.getenv("SELL_COST_BPS", "44.25")) / 10000
    cash, shares = 1.0, 0.0
    trades = wins = closed_trades = 0
    cycle_cost = cycle_proceeds = 0.0
    pending = None
    equity_curve = [1.0]
    cols = ["Close", "MA5", "MA10", "MA20", "RSI", "VOL20", "HIGH20_PREV"]

    # Execute yesterday's signal at today's open, mark today's close,
    # then decide tomorrow's order. The final close cannot create a fill.
    # Include every execution/signal input. Same-length revised snapshots and
    # copied DataFrames may carry old attrs; dates alone cannot validate them.
    fingerprint = pd.util.hash_pandas_object(
        df[list(dict.fromkeys(["Open", "Volume", *cols, *(["INSTITUTIONAL_2D"] if "INSTITUTIONAL_2D" in df else [])]))], index=True
    ).to_numpy().tobytes()
    cache_key = (START_DATE, fingerprint)
    if source.attrs.get("record_cache_key") != cache_key:
        source.attrs["record_cache"] = df.to_dict("records")
        source.attrs["record_cache_key"] = cache_key
    for row in source.attrs["record_cache"]:
        opening, close = float(row["Open"]), float(row["Close"])
        if pending == "buy":
            spent = cash
            shares = spent / (opening * (1 + buy_cost))
            cash = 0.0
            cycle_cost, cycle_proceeds = spent, 0.0
            trades += 1
        elif pending == "exit":
            proceeds = shares * opening * (1 - sell_cost)
            cash += proceeds
            cycle_proceeds += proceeds
            wins += int(cycle_proceeds > cycle_cost)
            closed_trades += 1
            shares = 0.0
        equity_curve.append(cash + shares * close)
        pending = None
        # Exit depends only on MA10; missing entry indicators cannot block it.
        if shares > 0:
            if math.isfinite(float(row["MA10"])) and close < row["MA10"]:
                pending = "exit"
        elif all(math.isfinite(float(row[c])) for c in cols) and entry_signal(row, p):
            pending = "buy"

    final_equity = equity_curve[-1]
    first_open, last_close = float(df["Open"].iloc[0]), float(df["Close"].iloc[-1])
    # Both portfolios remain marked at the final close, without a fictitious
    # sale. Also report comparable net-liquidation values separately.
    buy_hold_equity = last_close / (first_open * (1 + buy_cost))
    buy_hold_pct = (buy_hold_equity - 1) * 100
    arr = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(arr)
    return {
        "return_pct": (final_equity - 1) * 100,
        "mdd_pct": float(np.min((arr / peaks - 1) * 100)),
        "trades": trades,
        "closed_trades": closed_trades,
        "winning_trades": wins,
        "open_trades": int(shares > 0),
        "win_rate_pct": wins / closed_trades * 100 if closed_trades else 0.0,
        "buy_hold_pct": buy_hold_pct,
        "liquidation_return_pct": (cash + shares * last_close * (1 - sell_cost) - 1) * 100,
        "buy_hold_liquidation_pct": (buy_hold_equity * (1 - sell_cost) - 1) * 100,
    }

def load_history_best():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None

def _bounded(values, low, high, digits=2):
    return sorted({round(min(high, max(low, float(v))), digits) for v in values})

def parameter_grid(previous_params=None):
    # Each scheduled run explores a different neighborhood. The GitHub run
    # number changes the search radius, so hourly executions are not identical.
    phase = RUN_ID % 6
    if previous_params:
        rsi_min0 = previous_params.get("rsi_min", 47)
        rsi_max0 = previous_params.get("rsi_max", 72)
        vol0 = previous_params.get("vol_ratio", 1.0)
        br0 = previous_params.get("breakout_pct")
        br0 = -1.0 if br0 is None else br0
        rsi_mins = _bounded([rsi_min0 - 4 + phase, rsi_min0 - 2, rsi_min0, rsi_min0 + 2], 35, 65, 0)
        rsi_maxs = _bounded([rsi_max0 - 4, rsi_max0, rsi_max0 + 2 + phase], 55, 85, 0)
        vols = _bounded([vol0 - .25, vol0 - .1, vol0, vol0 + .15 + .05 * phase], .5, 2.0)
        breakouts = _bounded([br0 - 2, br0 - 1, br0, br0 + 1 + .5 * phase], -6, 5)
        breakouts = [None if abs(v + 1.0) < 1e-9 else v for v in breakouts]
    else:
        rsi_mins = [40, 45, 50, 55]
        rsi_maxs = [65, 70, 75, 80]
        vols = [0.7, 0.9, 1.1, 1.3]
        breakouts = [None, -3.0, -1.0, 0.0, 2.0]

    for ma_order, rsi_min, rsi_max, vol_ratio, breakout_pct in product(
        [False, True], rsi_mins, rsi_maxs, vols, breakouts
    ):
        if rsi_min >= rsi_max:
            continue
        for ma10_confirm in [1]:
            yield {
                "institutional_2d": True,
                "ma10_confirm": ma10_confirm,
                "ma_order": ma_order,
                "rsi_min": rsi_min,
                "rsi_max": rsi_max,
                "vol_ratio": vol_ratio,
                "breakout_pct": breakout_pct,
            }

def score(summary):
    # Primary objective: percentage of stocks individually exceeding +20%.
    # Tie-breakers penalize drawdown and reward median return.
    return (
        summary["pass_rate_pct"],
        summary.get("risk_pass_rate_pct", 0),
        summary["median_return_pct"],
        summary["avg_return_pct"],
        -abs(summary["avg_mdd_pct"]),
    )

def summarize(details, params, iteration=0):
    returns = [x["return_pct"] for x in details]
    closed = sum(x["closed_trades"] for x in details)
    passed = sum(x > TARGET_RETURN for x in returns)
    risk_passed = sum(x["return_pct"] > TARGET_RETURN and x["mdd_pct"] >= -20 for x in details)
    n = len(details)
    def mean(field):
        return float(np.mean([x[field] for x in details])) if n else 0.0
    return {
        "iteration": iteration, "symbols": n,
        "passed_symbols": passed, "pass_rate_pct": 100 * passed / n if n else 0,
        "risk_passed_symbols": risk_passed, "risk_pass_rate_pct": 100 * risk_passed / n if n else 0,
        "avg_trades": mean("trades"), "avg_win_rate_pct": mean("win_rate_pct"),
        "pooled_win_rate_pct": 100 * sum(x["winning_trades"] for x in details) / closed if closed else 0,
        "avg_buy_hold_pct": mean("buy_hold_pct"), "avg_return_pct": mean("return_pct"),
        "median_return_pct": float(np.median(returns)) if n else 0,
        "avg_mdd_pct": mean("mdd_pct"), "params": params,
    }

def evaluate(data, params, start_date=None, end_date=None, iteration=0):
    details = []
    for code, df in data.items():
        result = backtest_one(df, params, prepared=True, start_date=start_date, end_date=end_date)
        if result is not None:
            details.append({"code": code, **result})
    return summarize(details, params, iteration), details

def select_training_params(data, grid, start_date, end_date):
    candidates = []
    for iteration, params in enumerate(grid, 1):
        summary, _ = evaluate(data, params, start_date, end_date, iteration)
        if summary["symbols"]:
            candidates.append(summary)
    if not candidates:
        raise RuntimeError("No usable training-period results")
    best = max(candidates, key=score)
    return best["params"], best, candidates

def main():
    os.makedirs("backtest_results", exist_ok=True)
    codes = listed_twse_codes()
    selected = select_symbols(codes, MAX_SYMBOLS)
    snapshot_dir = "backtest_results/price_snapshot"
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f"[BACKTEST] TWSE only, symbols={len(selected)}, start={START_DATE}")
    data = {}
    for idx, code in enumerate(selected, 1):
        try:
            df = yf.download(code + ".TW", start="2025-09-01", auto_adjust=True, progress=False, timeout=20)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 80:
                # Preserve the exact research inputs in the Actions artifact.
                df.to_csv(f"{snapshot_dir}/{code}.csv", index_label="Date")
                data[code] = indicators(df)
            print(f"[{idx}/{len(selected)}] {code}: {len(df)}")
        except Exception as exc:
            print(f"[WARN] {code}: {exc}")

    if len(data) < 20:
        raise RuntimeError(f"Too few usable symbols: {len(data)}")

    calendar = sorted(set().union(*(set(df.index) for df in data.values())))
    sessions, reports, failures = load_reports(calendar, START_DATE)
    institutional_dir = "backtest_results/institutional_snapshot"
    os.makedirs(institutional_dir, exist_ok=True)
    for date, report in reports.items():
        with open(f"{institutional_dir}/{date}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
    coverage = {"source": "TWSE T86", "requested_sessions": len(sessions),
                "loaded_sessions": len(reports), "failed_sessions": failures}
    with open("backtest_results/institutional_coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)
    if failures:
        raise RuntimeError("Incomplete T86 history: no performance result published; see institutional_coverage.json")
    data = {code: attach_gate(df, code, sessions, reports) for code, df in data.items()}
    coverage["missing_stock_sessions"] = {code: int((~df.loc[df.index >= pd.Timestamp(START_DATE, tz=df.index.tz), "INSTITUTIONAL_AVAILABLE"]).sum()) for code, df in data.items()}

    # Fixed calendar windows and a fixed grid. Never use full-period winners
    # from earlier runs to seed training: they have already seen later prices.
    year = pd.Timestamp(START_DATE).year
    validation_start = f"{year}-05-01"
    later_start = f"{year}-08-01"
    if pd.Timestamp(START_DATE) >= pd.Timestamp(validation_start):
        raise ValueError("BACKTEST_START must precede the May validation boundary")
    best_params, training, candidates = select_training_params(
        data, list(parameter_grid()), START_DATE, validation_start
    )
    validation, validation_details = evaluate(data, best_params, validation_start, later_start)
    later, later_details = evaluate(data, best_params, later_start)
    run_best, best_details = evaluate(data, best_params)
    temporal_validation = {
        "selection": "fixed grid, training period only; parameters frozen for later windows",
        "training": {"start": START_DATE, "end_exclusive": validation_start, "summary": training},
        "validation": {"start": validation_start, "end_exclusive": later_start, "summary": validation, "stock_results": validation_details},
        "later_period": {"start": later_start, "summary": later, "stock_results": later_details},
        "later_period_status": "exploratory: dates were inspected by earlier full-period optimization; not a sealed holdout",
        "window_valuation": "each window starts in cash; indicators retain earlier warmup data; ending holdings marked at close",
    }
    improved_history = None  # Old snapshots are not comparable.
    historical_best = {
        "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "github_run_number": RUN_ID,
        "best": run_best,
        "best_params": best_params,
        "stock_results": sorted(best_details, key=lambda x: x["return_pct"], reverse=True),
    }

    descriptive_reached = run_best["pass_rate_pct"] >= TARGET_PASS_RATE
    # Previously inspected dates cannot establish an unseen-data target claim.
    reached = False
    output = {
        "engine_version": "7-ma10-institutional-two-day",
        "entry_scope": "technical entry AND foreign/trust each net-buy on both current and preceding market trading session; order next session open",
        "institutional_gate_backtested": True,
        "institutional_coverage": coverage,
        "institutional_timing": "T86 trade-date reports assumed available by next open; historical original release timestamps unavailable; missing stock records block entry; never forward-fill",
        "price_snapshot": "Actions artifact: price_snapshot/*.csv; exact adjusted OHLCV inputs",
        "price_basis": "Yahoo adjusted OHLC; corporate-action-adjusted research prices",
        "validation_status": "temporal_split_exploratory_not_sealed",
        "temporal_validation": temporal_validation,
        "descriptive_full_period_target_reached": descriptive_reached,
        "target_reached_reason": "No sealed unseen-data validation; full-period metrics are descriptive only",
        "data_end_date": max(str(df.index[-1].date()) for df in data.values()),
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "universe": "TWSE listed common stocks only",
        "sample_method": "all current TWSE codes" if len(selected) == len(codes) else "evenly spaced across code range with 2330 reserved; not sector-stratified or market-wide",
        "universe_symbols": len(codes),
        "selected_codes": selected,
        "universe_basis": "current listing membership; historical delistings excluded (survivorship bias possible)",
        "requested_symbols": len(selected),
        "downloaded_symbols": len(data),
        "failed_symbols": [code for code in selected if code not in data],
        "cost_bps": {"buy": float(os.getenv("BUY_COST_BPS", "14.25")), "sell": float(os.getenv("SELL_COST_BPS", "44.25"))},
        "valuation": "strategy and buy-and-hold marked at final close without synthetic sale; net-liquidation values separate; win rate uses fully closed cycles only",
        "start_date": START_DATE,
        "target": {
            "individual_return_pct": TARGET_RETURN,
            "required_pass_rate_pct": TARGET_PASS_RATE,
        },
        "exit_rules": {
            "below_ma5": "hold full position while close >= MA10",
            "below_ma10": "exit full position next open",
        },
        "github_run_number": RUN_ID,
        "improved_history": improved_history,
        "target_reached": reached,
        "run_best": run_best,
        "best": historical_best["best"],
        "best_params": historical_best["best_params"],
        "stock_results": historical_best["stock_results"],
        "top_candidates": sorted(candidates, key=score, reverse=True)[:20],
        "note": "Optimization target is not a promise of future returns. Validate on unseen periods before deployment.",
    }

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(historical_best, f, ensure_ascii=False, indent=2)

    with open("backtest_results/latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pd.DataFrame(output["stock_results"]).to_csv("backtest_results/latest_stocks.csv", index=False, encoding="utf-8-sig")
    with open("backtest_results/best_params.json", "w", encoding="utf-8") as f:
        json.dump(historical_best["best_params"], f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("BEST RESULT")
    print(json.dumps(historical_best["best"], ensure_ascii=False, indent=2))
    print(f"IMPROVED HISTORY: {improved_history}")
    print(f"TARGET REACHED: {reached}")

if __name__ == "__main__":
    main()
