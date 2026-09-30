# -*- coding: utf-8 -*-
"""Backtest MA trend strategy for Taiwan stocks.

Rules:
- Signals are computed at close and executed at next open.
- Exit is fixed: close < MA5 -> reduce to 50%; close < MA10 -> exit all.
- Entry variants change only the entry filter.
- Default test window starts 2026-01-01.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

COST_RATE = 0.001425 * 0.3
TAX_RATE = 0.003

def indicators(df):
    d=df.copy()
    for n in (5,10,20,60):
        d[f"MA{n}"]=d["Close"].rolling(n).mean()
    delta=d["Close"].diff()
    gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    d["RSI14"]=100-(100/(1+rs))
    d["VOL20"]=d["Volume"].rolling(20).mean()
    d["HIGH20"]=d["High"].shift(1).rolling(20).max()
    d["MA20_UP"]=d["MA20"]>d["MA20"].shift(5)
    return d

def entry_signal(row, variant):
    c,m5,m10,m20=row["Close"],row["MA5"],row["MA10"],row["MA20"]
    if any(pd.isna(x) for x in (c,m5,m10,m20)): return False
    base=c>m5 and c>m10 and c>m20
    trend=base and m5>m10>m20
    if variant=="baseline": return base
    if variant=="trend": return trend
    if variant=="trend_slope": return trend and bool(row["MA20_UP"])
    if variant=="trend_rsi": return trend and 45 <= row["RSI14"] <= 70
    if variant=="trend_volume": return trend and row["Volume"] >= row["VOL20"]*1.2
    if variant=="trend_breakout": return trend and c >= row["HIGH20"]*0.99
    if variant=="combo":
        return trend and bool(row["MA20_UP"]) and 45 <= row["RSI14"] <= 72 and row["Volume"] >= row["VOL20"]
    raise ValueError(variant)

def backtest(df, variant, start):
    d=indicators(df).dropna(subset=["Open","High","Low","Close"]).copy()
    d=d.loc[d.index>=pd.Timestamp(start)]
    if len(d)<3: return None
    cash=1.0; shares=0.0; state=0; trades=0; wins=0; entry_value=0.0; realized=0.0
    equity=[]
    for i in range(len(d)-1):
        r=d.iloc[i]; nxt=d.iloc[i+1]; px=float(nxt["Open"])
        if shares==0 and entry_signal(r,variant):
            shares=(cash*(1-COST_RATE))/px; cash=0; state=2; trades+=1
            entry_value=shares*px; realized=0.0
        elif shares>0:
            if r["Close"] < r["MA10"]:
                proceeds=shares*px*(1-COST_RATE-TAX_RATE)
                if proceeds + realized > entry_value: wins+=1
                cash+=proceeds; shares=0; state=0
            elif r["Close"] < r["MA5"] and state==2:
                sell=shares*0.5
                part=sell*px*(1-COST_RATE-TAX_RATE); cash+=part; realized+=part; shares-=sell; state=1
        equity.append(cash+shares*float(r["Close"]))
    if shares>0:\n        mark=shares*float(d.iloc[-1]["Close"])*(1-COST_RATE-TAX_RATE)\n        if mark + realized > entry_value: wins+=1\n    final=cash+shares*float(d.iloc[-1]["Close"])
    eq=pd.Series(equity+[final])
    peak=eq.cummax()
    mdd=((eq/peak)-1).min()
    bh=float(d.iloc[-1]["Close"]/d.iloc[0]["Open"]-1)
    return {"return_pct":round((final-1)*100,2),"mdd_pct":round(mdd*100,2),
            "trades":trades,"win_rate_pct":round(wins/max(trades,1)*100,2),
            "buy_hold_pct":round(bh*100,2)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2026-01-01")
    ap.add_argument("--codes",default="2330,2317,2382,2454,3711,3231,6669,2303,3034,2379")
    ap.add_argument("--out",default="data/backtest_results.csv")
    a=ap.parse_args()
    codes=[x.strip() for x in a.codes.split(",") if x.strip()]
    variants=["baseline","trend","trend_slope","trend_rsi","trend_volume","trend_breakout","combo"]
    rows=[]
    for code in codes:
        try:
            raw=yf.download(code+".TW",start="2025-09-01",auto_adjust=False,progress=False)
            if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
            for v in variants:
                r=backtest(raw,v,a.start)
                if r: rows.append({"code":code,"variant":v,**r})
        except Exception as e:
            print("[WARN]",code,e)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False,encoding="utf-8-sig")
    summary=(pd.DataFrame(rows).groupby("variant").agg(
        avg_return_pct=("return_pct","mean"), median_return_pct=("return_pct","median"),
        avg_mdd_pct=("mdd_pct","mean"), avg_win_rate_pct=("win_rate_pct","mean"),
        stocks=("code","count")).sort_values("avg_return_pct",ascending=False))
    print(summary.to_string())
    Path("data/backtest_summary.json").write_text(summary.reset_index().to_json(orient="records",force_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
