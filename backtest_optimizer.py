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
    trades = 0
    wins = 0
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
                entry_value = shares * next_open
                trades += 1
        else:
            if close < float(r.MA10):
                exit_value = shares * next_open
                if exit_value > entry_value:
                    wins += 1
                cash += exit_value
                shares = 0.0
                reduced = False
            elif close < float(r.MA5) and not reduced:
                sell = shares * 0.5
                cash += sell * next_open
                shares -= sell
                reduced = True

        equity_curve.append(cash + shares * close)

    last_close = float(df["Close"].iloc[-1])
    final_equity = cash + shares * last_close
    arr = np.asarray(equity_curve or [1.0], dtype=float)
    peaks = np.maximum.accumulate(arr)
    mdd = float(np.min((arr / peaks - 1.0) * 100))
    return {
        "return_pct": (final_equity - 1.0) * 100,
        "mdd_pct": mdd,
        "trades": trades,
        "win_rate_pct": (wins / trades * 100) if trades else 0.0,
    }

def parameter_grid():
    # Deliberately bounded grid to keep scheduled Actions runtime predictable.
    for ma_order, rsi_min, rsi_max, vol_ratio, breakout_pct in product(
        [False, True],
        [45, 50],
        [68, 72, 76],
        [0.8, 1.0, 1.2],
        [None, -2.0, 0.0],
    ):
        yield {
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

    candidates = []
    best_details = None
    best_summary = None
    best_params = None

    for iteration, params in enumerate(parameter_grid(), 1):
        details = []
        for code, df in data.items():
            result = backtest_one(df, params)
            if result is not None:
                details.append({"code": code, **result})

        returns = [x["return_pct"] for x in details]
        mdds = [x["mdd_pct"] for x in details]
        passed = sum(x >= TARGET_RETURN for x in returns)
        summary = {
            "iteration": iteration,
            "symbols": len(details),
            "passed_symbols": passed,
            "pass_rate_pct": 100 * passed / len(details) if details else 0,
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

    reached = best_summary["pass_rate_pct"] >= TARGET_PASS_RATE
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
        "target_reached": reached,
        "best": best_summary,
        "best_params": best_params,
        "stock_results": sorted(best_details, key=lambda x: x["return_pct"], reverse=True),
        "top_candidates": sorted(candidates, key=score, reverse=True)[:20],
        "note": "Optimization target is not a promise of future returns. Validate on unseen periods before deployment.",
    }

    with open("backtest_results/latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pd.DataFrame(output["stock_results"]).to_csv("backtest_results/latest_stocks.csv", index=False, encoding="utf-8-sig")
    with open("backtest_results/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("BEST RESULT")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))
    print(f"TARGET REACHED: {reached}")

if __name__ == "__main__":
    main()
