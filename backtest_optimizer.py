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
# close < MA5 => reduce position by 50%
# close < MA10 => exit remaining position
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
    x["RSI"] = 100 - 100 / (1 + rs)
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["HIGH20_PREV"] = x["High"].shift(1).rolling(20).max()
    return x.dropna().copy()

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

def backtest_one(df, p):
    df = indicators(df)
    df = df[df.index >= pd.Timestamp(START_DATE, tz=df.index.tz) if df.index.tz is not None else df.index >= pd.Timestamp(START_DATE)]
    if len(df) < 25:
        return None

    cash = 1.0
    shares = 0.0
    reduced = False
    below5_days = 0
    below10_days = 0
    trades = 0
    wins = 0
    closed_trades = 0
    realized_proceeds = 0.0
    equity_curve = []

    rows = list(df.itertuples())
    for i in range(len(rows) - 1):
        r = rows[i]
        nxt = rows[i + 1]
        close = float(r.Close)
        next_open = float(nxt.Open)

        if shares == 0:
            row = df.iloc[i]
            if entry_signal(row, p):
                shares = cash / next_open
                cash = 0.0
                reduced = False
                below5_days = 0
                below10_days = 0
                entry_value = shares * next_open
                realized_proceeds = 0.0
                trades += 1
        else:
            below5_days = below5_days + 1 if close < float(r.MA5) else 0
            below10_days = below10_days + 1 if close < float(r.MA10) else 0

            if below10_days >= p["ma10_confirm"]:
                exit_value = shares * next_open
                total_proceeds = realized_proceeds + exit_value
                if total_proceeds > entry_value:
                    wins += 1
                closed_trades += 1
                cash += exit_value
                shares = 0.0
                reduced = False
                below5_days = 0
                below10_days = 0
            elif below5_days >= p["ma5_confirm"] and not reduced:
                sell = shares * 0.5
                proceeds = sell * next_open
                cash += proceeds
                realized_proceeds += proceeds
                shares -= sell
                reduced = True
            elif p["reentry"] and reduced and close > float(r.MA5) and close > float(r.MA10):
                # Restore the reduced half only after a close confirms recovery;
                # execution remains at next open, so no future price is used.
                add_value = min(cash, shares * next_open)
                if add_value > 0:
                    shares += add_value / next_open
                    cash -= add_value
                    entry_value = shares * next_open
                    realized_proceeds = 0.0
                    reduced = False
                    below5_days = 0
                    below10_days = 0

        equity_curve.append(cash + shares * close)

    last_close = float(df["Close"].iloc[-1])
    final_equity = cash + shares * last_close
    if shares > 0:
        total_proceeds = realized_proceeds + shares * last_close
        if total_proceeds > entry_value:
            wins += 1
        closed_trades += 1
    first_open = float(df["Open"].iloc[0])
    buy_hold_pct = (last_close / first_open - 1.0) * 100 if first_open > 0 else 0.0
    arr = np.asarray(equity_curve or [1.0], dtype=float)
    peaks = np.maximum.accumulate(arr)
    mdd = float(np.min((arr / peaks - 1.0) * 100))
    return {
        "return_pct": (final_equity - 1.0) * 100,
        "mdd_pct": mdd,
        "trades": trades,
        "win_rate_pct": (wins / closed_trades * 100) if closed_trades else 0.0,
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
        for ma5_confirm, ma10_confirm, reentry in product([1, 2, 3], [1, 2, 3], [False, True]):
            yield {
                "ma5_confirm": ma5_confirm,
                "ma10_confirm": ma10_confirm,
                "reentry": reentry,
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
    # Deterministic liquid-ish sample by availability; 2330 is always included.
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
                data[code] = df
            print(f"[{idx}/{len(selected)}] {code}: {len(df)}")
        except Exception as exc:
            print(f"[WARN] {code}: {exc}")

    if len(data) < 20:
        raise RuntimeError(f"Too few usable symbols: {len(data)}")

    history_best = load_history_best()
    previous_params = history_best.get("best_params") if history_best else None
    candidates = []
    best_details = None
    best_summary = None
    best_params = None

    for iteration, params in enumerate(parameter_grid(previous_params), 1):
        details = []
        for code, df in data.items():
            result = backtest_one(df, params)
            if result is not None:
                details.append({"code": code, **result})

        returns = [x["return_pct"] for x in details]
        mdds = [x["mdd_pct"] for x in details]
        passed = sum(x >= TARGET_RETURN for x in returns)
        risk_passed = sum(x["return_pct"] >= TARGET_RETURN and x["mdd_pct"] >= -20.0 for x in details)
        summary = {
            "iteration": iteration,
            "symbols": len(details),
            "passed_symbols": passed,
            "pass_rate_pct": 100 * passed / len(details) if details else 0,
            "risk_passed_symbols": risk_passed,
            "risk_pass_rate_pct": 100 * risk_passed / len(details) if details else 0,
            "avg_trades": float(np.mean([x["trades"] for x in details])) if details else 0,
            "avg_win_rate_pct": float(np.mean([x["win_rate_pct"] for x in details])) if details else 0,
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
    improved_history = history_best is None or score(run_best) > score(history_best["best"])
    if history_best is not None and not improved_history:
        historical_best = history_best
    else:
        historical_best = {
            "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
            "github_run_number": RUN_ID,
            "best": run_best,
            "best_params": best_params,
            "stock_results": sorted(best_details, key=lambda x: x["return_pct"], reverse=True),
        }

    reached = historical_best["best"]["pass_rate_pct"] >= TARGET_PASS_RATE
    output = {
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "universe": "TWSE listed common stocks only",
        "start_date": START_DATE,
        "target": {
            "individual_return_pct": TARGET_RETURN,
            "required_pass_rate_pct": TARGET_PASS_RATE,
        },
        "exit_rules": {
            "below_ma5": "reduce 50% next open",
            "below_ma10": "exit remaining position next open",
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
