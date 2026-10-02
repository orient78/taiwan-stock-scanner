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

def entry_signal(row, p):
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

def backtest_one(df, p, prepared=False):
    if p.get("ma10_confirm", 1) != 1:
        raise ValueError("MA10 exit must execute after one close below the MA")
    df = df if prepared else indicators(df)
    source = df
    start = pd.Timestamp(START_DATE, tz=df.index.tz)
    df = df[df.index >= start]
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
    cache_key = (START_DATE, len(df), str(df.index[-1]))
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
    buy_hold_pct = (last_close * (1 - sell_cost) / (first_open * (1 + buy_cost)) - 1) * 100
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

def main():
    os.makedirs("backtest_results", exist_ok=True)
    codes = listed_twse_codes()
    # Deterministic code-order sample, not a liquidity or market-wide sample.
    selected = codes[:MAX_SYMBOLS]
    if "2330" not in selected:
        selected[-1] = "2330"

    print(f"[BACKTEST] TWSE only, symbols={len(selected)}, start={START_DATE}")
    data = {}
    for idx, code in enumerate(selected, 1):
        try:
            df = yf.download(code + ".TW", start="2025-09-01", auto_adjust=False, progress=False, timeout=20)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 80:
                data[code] = indicators(df)
            print(f"[{idx}/{len(selected)}] {code}: {len(df)}")
        except Exception as exc:
            print(f"[WARN] {code}: {exc}")

    if len(data) < 20:
        raise RuntimeError(f"Too few usable symbols: {len(data)}")

    # Historical metrics from different dates/data must never compete with
    # fresh metrics. Re-evaluate the incumbent on the current snapshot.
    history_best = load_history_best()
    previous_params = history_best.get("best_params") if history_best else None
    candidates = []
    best_details = None
    best_summary = None
    best_params = None

    grid = list(parameter_grid(previous_params))
    if previous_params:
        incumbent = {k: v for k, v in previous_params.items() if k not in ("ma5_confirm", "reentry")}
        incumbent["ma10_confirm"] = 1
        if incumbent not in grid:
            grid.insert(0, incumbent)
    for iteration, params in enumerate(grid, 1):
        details = []
        for code, df in data.items():
            result = backtest_one(df, params, prepared=True)
            if result is not None:
                details.append({"code": code, **result})

        returns = [x["return_pct"] for x in details]
        mdds = [x["mdd_pct"] for x in details]
        passed = sum(x > TARGET_RETURN for x in returns)
        risk_passed = sum(x["return_pct"] > TARGET_RETURN and x["mdd_pct"] >= -20.0 for x in details)
        summary = {
            "iteration": iteration,
            "symbols": len(details),
            "passed_symbols": passed,
            "pass_rate_pct": 100 * passed / len(details) if details else 0,
            "risk_passed_symbols": risk_passed,
            "risk_pass_rate_pct": 100 * risk_passed / len(details) if details else 0,
            "avg_trades": float(np.mean([x["trades"] for x in details])) if details else 0,
            "avg_win_rate_pct": float(np.mean([x["win_rate_pct"] for x in details])) if details else 0,
            "pooled_win_rate_pct": 100 * sum(x["winning_trades"] for x in details) / sum(x["closed_trades"] for x in details) if sum(x["closed_trades"] for x in details) else 0,
            "avg_buy_hold_pct": float(np.mean([x["buy_hold_pct"] for x in details])) if details else 0,
            "avg_return_pct": float(np.mean(returns)) if returns else 0,
            "median_return_pct": float(np.median(returns)) if returns else 0,
            "avg_mdd_pct": float(np.mean(mdds)) if mdds else 0,
            "params": params,
        }
        candidates.append(summary)
        if best_summary is None or score(summary) > score(best_summary):
            best_summary = summary
            best_params = params
            best_details = details

    run_best = best_summary
    improved_history = None  # Old snapshots are not comparable.
    historical_best = {
        "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "github_run_number": RUN_ID,
        "best": run_best,
        "best_params": best_params,
        "stock_results": sorted(best_details, key=lambda x: x["return_pct"], reverse=True),
    }

    reached = historical_best["best"]["pass_rate_pct"] >= TARGET_PASS_RATE
    output = {
        "engine_version": "4-ma10-full-position",
        "validation_status": "in_sample_only",
        "data_end_date": max(str(df.index[-1].date()) for df in data.values()),
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "universe": "TWSE listed common stocks only",
        "sample_method": "first MAX_SYMBOLS by code, with 2330 forced in; not market-wide",
        "requested_symbols": len(selected),
        "downloaded_symbols": len(data),
        "failed_symbols": [code for code in selected if code not in data],
        "cost_bps": {"buy": float(os.getenv("BUY_COST_BPS", "14.25")), "sell": float(os.getenv("SELL_COST_BPS", "44.25"))},
        "valuation": "open holdings marked at final close; win rate uses fully closed cycles only",
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

