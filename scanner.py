import os
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import twstock

from stock_pool import get_stock_codes, get_groups


# ============================================================
# V6 CONFIG
# ============================================================

VERSION = "V6"
TIMEZONE = "Asia/Taipei"

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000

CHART_DAYS = 120

# 首頁各雷達數量
NEXT_DAY_TOP = 10
READY_TOP = 20
MID_LONG_TOP = 20
RADAR_TOP = 80

ETF_CODES = {
    "0050", "006208", "0052", "0053",
    "00881", "00891", "00927", "00935",
    "0056", "00878", "00919", "00929", "00940",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def r2(value):
    return round(safe_float(value), 2)


def pct(a, b):
    if not b:
        return 0.0
    return (a / b - 1.0) * 100.0


# ============================================================
# 中文名稱 / 市場
# ============================================================

def get_stock_info(code):
    """
    使用 twstock 本地代碼表。
    避免 Yahoo Finance 回傳英文公司名稱。
    """
    try:
        info = twstock.codes.get(code)

        if info:
            return {
                "name": info.name,
                "market": info.market,
                "industry": getattr(info, "group", "") or "",
            }
    except Exception:
        pass

    return {
        "name": code,
        "market": "",
        "industry": "",
    }


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def add_indicators(df):
    df = df.copy()

    # Moving averages
    for period in [5, 10, 20, 60, 120, 240]:
        df[f"MA{period}"] = df["Close"].rolling(period).mean()

    # RSI - only used as overheat filter
    df["RSI14"] = calculate_rsi(df["Close"])

    # Volume
    df["VOL5"] = df["Volume"].rolling(5).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    # Breakout
    df["HIGH20"] = df["High"].rolling(20).max().shift(1)
    df["HIGH60"] = df["High"].rolling(60).max().shift(1)

    # ATR
    df["ATR14"] = calculate_atr(df)

    # Recent lows / support
    df["LOW10"] = df["Low"].rolling(10).min().shift(1)
    df["LOW20"] = df["Low"].rolling(20).min().shift(1)
    df["LOW60"] = df["Low"].rolling(60).min().shift(1)

    # Returns
    df["RET5"] = df["Close"].pct_change(5) * 100
    df["RET20"] = df["Close"].pct_change(20) * 100
    df["RET60"] = df["Close"].pct_change(60) * 100
    df["RET120"] = df["Close"].pct_change(120) * 100

    return df


# ============================================================
# MARKET DATA
# ============================================================

def download_history(code):
    """
    先嘗試上市 .TW，再嘗試上櫃 .TWO
    """

    candidates = [
        (f"{code}.TW", "上市"),
        (f"{code}.TWO", "上櫃"),
    ]

    for ticker, market in candidates:
        try:
            df = yf.download(
                ticker,
                period="18mo",
                interval="1d",
                auto_adjust=True,
                repair=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = {"Open", "High", "Low", "Close", "Volume"}

            if not required.issubset(df.columns):
                continue

            df = df.dropna(subset=["Open", "High", "Low", "Close"])

            if len(df) >= 80:
                return ticker, market, df

        except Exception as e:
            print(f"[WARN] {ticker}: {e}")

    return None, None, None


# ============================================================
# CHART
# ============================================================

def build_chart_data(df):
    out = []

    use = df.tail(CHART_DAYS)

    for idx, row in use.iterrows():
        out.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": r2(row["Open"]),
                "high": r2(row["High"]),
                "low": r2(row["Low"]),
                "close": r2(row["Close"]),
                "volume": int(safe_float(row["Volume"])),
                "ma5": r2(row.get("MA5")),
                "ma10": r2(row.get("MA10")),
                "ma20": r2(row.get("MA20")),
                "ma60": r2(row.get("MA60")),
            }
        )

    return out


# ============================================================
# MARKET RELATIVE STRENGTH
# ============================================================

def get_market_reference():
    """
    台灣加權指數，用來計算 Relative Strength。
    """

    try:
        df = yf.download(
            "^TWII",
            period="18mo",
            interval="1d",
            auto_adjust=True,
            repair=True,
            progress=False,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df is None or len(df) < 60:
            return {}

        close = df["Close"]

        return {
            "ret5": safe_float(close.pct_change(5).iloc[-1] * 100),
            "ret20": safe_float(close.pct_change(20).iloc[-1] * 100),
            "ret60": safe_float(close.pct_change(60).iloc[-1] * 100),
            "ma20": safe_float(close.rolling(20).mean().iloc[-1]),
            "ma60": safe_float(close.rolling(60).mean().iloc[-1]),
            "price": safe_float(close.iloc[-1]),
        }

    except Exception as e:
        print("[WARN] Market reference failed:", e)
        return {}


# ============================================================
# CORE V6 SCORES
# ============================================================

def calculate_trend_score(row):
    """
    25 points
    """

    score = 0
    reasons = []

    price = safe_float(row["Close"])
    ma5 = safe_float(row["MA5"])
    ma10 = safe_float(row["MA10"])
    ma20 = safe_float(row["MA20"])
    ma60 = safe_float(row["MA60"])

    if price > ma20:
        score += 6
        reasons.append("股價站上20日均線")

    if ma5 > ma10 > ma20:
        score += 8
        reasons.append("MA5 > MA10 > MA20，多頭排列")

    if ma20 > ma60:
        score += 6
        reasons.append("MA20位於MA60之上")

    if ma20 > 0:
        score += 5
        reasons.append("中短期趨勢維持偏多")

    return min(score, 25), reasons


def calculate_breakout_score(row):
    """
    25 points
    """

    price = safe_float(row["Close"])
    high20 = safe_float(row["HIGH20"])

    if not high20:
        return 0, [], 999

    distance = (high20 - price) / high20 * 100

    score = 0
    reasons = []

    if price >= high20:
        score = 25
        reasons.append("已突破20日高點")

    elif distance <= 1:
        score = 23
        reasons.append("距20日突破價不到1%")

    elif distance <= 2:
        score = 21
        reasons.append("非常接近20日突破")

    elif distance <= 4:
        score = 17
        reasons.append("接近20日突破區")

    elif distance <= 7:
        score = 10
        reasons.append("距突破位置尚可")

    return score, reasons, distance


def calculate_volume_score(row):
    """
    20 points
    """

    volume = safe_float(row["Volume"])
    vol20 = safe_float(row["VOL20"])

    ratio = volume / vol20 if vol20 else 0

    score = 0
    reasons = []

    if ratio >= 2:
        score = 20
        reasons.append("成交量超過20日均量2倍")

    elif ratio >= 1.5:
        score = 18
        reasons.append("成交量明顯放大")

    elif ratio >= 1.2:
        score = 15
        reasons.append("量能開始擴張")

    elif ratio >= 1:
        score = 10
        reasons.append("成交量高於20日均量")

    elif ratio >= 0.75:
        score = 5

    return score, ratio, reasons


def calculate_rs_score(row, market):
    """
    Relative Strength
    15 points
    """

    stock20 = safe_float(row["RET20"])
    stock60 = safe_float(row["RET60"])

    market20 = safe_float(market.get("ret20"))
    market60 = safe_float(market.get("ret60"))

    rs20 = stock20 - market20
    rs60 = stock60 - market60

    score = 0
    reasons = []

    if rs20 >= 10:
        score += 9
    elif rs20 >= 5:
        score += 7
    elif rs20 >= 0:
        score += 5
    elif rs20 >= -3:
        score += 2

    if rs60 >= 15:
        score += 6
    elif rs60 >= 8:
        score += 5
    elif rs60 >= 0:
        score += 3

    if rs20 > 0:
        reasons.append(f"近20日表現優於大盤 {rs20:.1f}%")

    return min(score, 15), rs20, rs60, reasons


# ============================================================
# SECTOR STRENGTH
# ============================================================

def build_sector_strength(stocks):
    groups = {}

    for stock in stocks:
        for group in stock["groups"]:
            groups.setdefault(group, []).append(stock)

    results = {}

    for group, members in groups.items():

        if not members:
            continue

        rs_values = [x["rs20"] for x in members]
        trend_values = [
            1 if x["price"] > x["ma20"] else 0
            for x in members
        ]
        ret20_values = [x["ret20"] for x in members]

        avg_rs = np.mean(rs_values)
        avg_ret20 = np.mean(ret20_values)
        trend_ratio = np.mean(trend_values)

        raw = (
            avg_rs * 0.45
            + avg_ret20 * 0.25
            + trend_ratio * 20 * 0.30
        )

        results[group] = {
            "group": group,
            "raw_score": safe_float(raw),
            "avg_rs20": r2(avg_rs),
            "avg_ret20": r2(avg_ret20),
            "trend_ratio": r2(trend_ratio * 100),
            "count": len(members),
        }

    ordered = sorted(
        results.values(),
        key=lambda x: x["raw_score"],
        reverse=True,
    )

    for rank, item in enumerate(ordered, 1):
        item["rank"] = rank

        if len(ordered) <= 1:
            item["score"] = 100
        else:
            percentile = 1 - ((rank - 1) / (len(ordered) - 1))
            item["score"] = round(percentile * 100)

    return {
        item["group"]: item
        for item in ordered
    }


def apply_sector_score(stock, sector_map):
    best = None

    for group in stock["groups"]:
        data = sector_map.get(group)

        if not data:
            continue

        if best is None or data["score"] > best["score"]:
            best = data

    if best is None:
        return 0, "", None

    # V6 sector component max = 15
    sector_component = round(best["score"] / 100 * 15)

    return sector_component, best["group"], best


# ============================================================
# ENTRY / STOP LOSS ENGINE
# ============================================================

def calculate_trade_plan(stock):
    """
    產生：
    - pullback entry
    - breakout trigger
    - chase limit
    - short stop
    - mid/long stop
    - target
    - R/R
    """

    price = stock["price"]
    ma10 = stock["ma10"]
    ma20 = stock["ma20"]
    ma60 = stock["ma60"]

    high20 = stock["high20"]
    atr = stock["atr14"]

    low10 = stock["low10"]
    low20 = stock["low20"]
    low60 = stock["low60"]

    # ----------------------------
    # Pullback entry zone
    # ----------------------------

    supports = [
        x for x in [ma10, ma20]
        if x > 0 and x <= price * 1.03
    ]

    if supports:
        support = max(supports)
    else:
        support = price * 0.97

    entry_low = max(
        support * 0.99,
        price - atr * 0.8 if atr else price * 0.97,
    )

    entry_high = min(
        price,
        support * 1.015,
    )

    if entry_low > entry_high:
        entry_low = price * 0.98
        entry_high = price

    # ----------------------------
    # Breakout trigger
    # ----------------------------

    breakout = high20 if high20 > 0 else price

    # ----------------------------
    # Chase limit
    # ----------------------------

    chase_by_pct = breakout * 1.03

    if atr:
        chase_by_atr = breakout + atr * 0.75
        chase_limit = min(chase_by_pct, chase_by_atr)
    else:
        chase_limit = chase_by_pct

    # ----------------------------
    # Short-term stop
    # ----------------------------

    stop_candidates = []

    if low10 > 0:
        stop_candidates.append(low10 * 0.995)

    if ma20 > 0:
        stop_candidates.append(ma20 * 0.985)

    if atr:
        stop_candidates.append(entry_low - atr * 1.5)

    valid_short = [
        x for x in stop_candidates
        if 0 < x < entry_low
    ]

    if valid_short:
        short_stop = max(valid_short)
    else:
        short_stop = entry_low * 0.94

    # Stop不能離買入區過近
    max_stop = entry_low * 0.985
    short_stop = min(short_stop, max_stop)

    # ----------------------------
    # Mid / long-term stop
    # ----------------------------

    long_candidates = []

    if ma60 > 0:
        long_candidates.append(ma60 * 0.97)

    if low60 > 0:
        long_candidates.append(low60 * 0.99)

    if atr:
        long_candidates.append(price - atr * 3)

    valid_long = [
        x for x in long_candidates
        if 0 < x < price
    ]

    if valid_long:
        long_stop = max(valid_long)
    else:
        long_stop = price * 0.88

    # ----------------------------
    # Target / R:R
    # ----------------------------

    risk = entry_high - short_stop

    if risk > 0:
        target1 = entry_high + risk * 2
        target2 = entry_high + risk * 3
        rr = 2.0
    else:
        target1 = price * 1.08
        target2 = price * 1.12
        rr = 0

    short_risk_pct = (
        (entry_high - short_stop) / entry_high * 100
        if entry_high else 0
    )

    long_risk_pct = (
        (price - long_stop) / price * 100
        if price else 0
    )

    return {
        "entry_low": r2(entry_low),
        "entry_high": r2(entry_high),
        "breakout_price": r2(breakout),
        "chase_limit": r2(chase_limit),

        "short_stop": r2(short_stop),
        "long_stop": r2(long_stop),

        "short_risk_pct": r2(short_risk_pct),
        "long_risk_pct": r2(long_risk_pct),

        "target1": r2(target1),
        "target2": r2(target2),

        "risk_reward": r2(rr),
    }


# ============================================================
# NEXT DAY SCORE
# ============================================================

def calculate_next_day_score(stock):
    """
    明日進場分數
    0-100

    Trend      25
    Breakout   25
    Volume     20
    RS         15
    Sector     15
    """

    score = (
        stock["trend_score"]
        + stock["breakout_score"]
        + stock["volume_score"]
        + stock["rs_score"]
        + stock["sector_score"]
    )

    # RSI only as overheat filter
    rsi = stock["rsi"]

    if rsi >= 80:
        score -= 25
    elif rsi >= 75:
        score -= 15
    elif rsi >= 72:
        score -= 7

    # Price too far from MA20
    distance_ma20 = stock["distance_ma20_pct"]

    if distance_ma20 >= 15:
        score -= 20
    elif distance_ma20 >= 12:
        score -= 12
    elif distance_ma20 >= 9:
        score -= 5

    return max(0, min(round(score), 100))


# ============================================================
# READY SCORE
# ============================================================

def calculate_ready_score(stock):
    """
    準備進場：
    還沒完全突破，但位置接近、趨勢健康。
    """

    score = 0

    if stock["price"] > stock["ma20"]:
        score += 20

    if stock["ma5"] > stock["ma10"] > stock["ma20"]:
        score += 20

    d = stock["breakout_distance_pct"]

    if 0 < d <= 1:
        score += 25
    elif d <= 2:
        score += 22
    elif d <= 4:
        score += 18
    elif d <= 7:
        score += 10

    if 0.7 <= stock["volume_ratio"] <= 1.5:
        score += 10

    if stock["rs20"] > 0:
        score += 15

    if stock["sector_score"] >= 10:
        score += 10

    if stock["rsi"] >= 75:
        score -= 15

    return max(0, min(round(score), 100))


# ============================================================
# MID/LONG SCORE
# ============================================================

def calculate_mid_long_score(stock):
    """
    中長期趨勢分數。
    目前 V6 先用 price trend + RS + sector。
    基本面資料之後可在 V6.1 加入。
    """

    score = 0

    price = stock["price"]

    if price > stock["ma20"]:
        score += 10

    if stock["ma20"] > stock["ma60"]:
        score += 20

    if stock["ma60"] > stock["ma120"] > 0:
        score += 20

    if stock["ma120"] > stock["ma240"] > 0:
        score += 15

    if stock["ret60"] > 0:
        score += 10

    if stock["rs60"] > 0:
        score += 15

    if stock["sector_score"] >= 10:
        score += 10

    return max(0, min(round(score), 100))


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_next_day(stock):
    score = stock["next_day_score"]

    if (
        stock["rsi"] >= 75
        or stock["distance_ma20_pct"] >= 12
    ):
        return "過熱／不追價"

    if (
        score >= 80
        and stock["breakout_distance_pct"] <= 2
        and stock["volume_ratio"] >= 1.2
    ):
        return "明日進場候選"

    if score >= 68:
        return "等待明日確認"

    return "暫不考慮"


def classify_ready(stock):
    score = stock["ready_score"]

    if stock["rsi"] >= 75:
        return "過熱"

    if score >= 80:
        return "準備進場"

    if score >= 65:
        return "持續觀察"

    return "尚未成熟"


def classify_mid_long(stock):
    score = stock["mid_long_score"]

    if score >= 80:
        return "中長期趨勢強"

    if score >= 65:
        return "中長期持續追蹤"

    return "中長期一般"


# ============================================================
# REASONS
# ============================================================

def build_reasons(stock):
    reasons = []
    risks = []

    if stock["ma5"] > stock["ma10"] > stock["ma20"]:
        reasons.append("短期均線呈多頭排列")

    if stock["ma20"] > stock["ma60"]:
        reasons.append("中期趨勢維持向上")

    if stock["breakout_distance_pct"] <= 2:
        reasons.append("已非常接近20日突破位置")

    elif stock["breakout_distance_pct"] <= 5:
        reasons.append("正在接近20日壓力區")

    if stock["volume_ratio"] >= 1.5:
        reasons.append(
            f"成交量放大至20日均量 {stock['volume_ratio']:.2f} 倍"
        )

    elif stock["volume_ratio"] >= 1.2:
        reasons.append("成交量開始擴張")

    if stock["rs20"] > 0:
        reasons.append(
            f"近20日相對大盤強 {stock['rs20']:.1f}%"
        )

    if stock["best_sector"]:
        reasons.append(
            f"{stock['best_sector']}族群相對強勢"
        )

    if stock["rsi"] >= 75:
        risks.append("RSI進入過熱區，不適合追價")

    elif stock["rsi"] >= 70:
        risks.append("RSI偏高，注意隔日追價風險")

    if stock["distance_ma20_pct"] >= 12:
        risks.append("股價與MA20乖離過大")

    if stock["volume_ratio"] < 0.8:
        risks.append("目前量能仍不足")

    if stock["breakout_distance_pct"] > 7:
        risks.append("距離突破位置仍較遠")

    if not reasons:
        reasons.append("目前以技術結構觀察為主")

    if not risks:
        risks.append("仍需觀察隔日開盤與成交量確認")

    return reasons[:5], risks[:4]


# ============================================================
# NEXT DAY ACTION
# ============================================================

def build_next_day_action(stock):
    p = stock["trade_plan"]

    breakout = p["breakout_price"]
    chase = p["chase_limit"]

    return (
        f"明日若突破 {breakout:.2f} 且量能同步放大，"
        f"可視為進場觸發；"
        f"若直接跳空高於 {chase:.2f}，不建議追價。"
    )


# ============================================================
# SCAN ONE
# ============================================================

def scan_one(code, market_ref):
    ticker, detected_market, df = download_history(code)

    if df is None:
        return None

    df = add_indicators(df)

    row = df.iloc[-1]

    price = safe_float(row["Close"])

    if price <= 0:
        return None

    is_etf = code in ETF_CODES

    avg_value = safe_float(
        (df["Close"] * df["Volume"]).tail(20).mean()
    )

    # 基本流動性過濾
    if not is_etf:
        if price < MIN_PRICE:
            return None

        if avg_value < MIN_AVG_DAILY_VALUE:
            return None

    info = get_stock_info(code)

    market = info["market"] or detected_market

    groups = get_groups(code)

    # ----------------------------
    # Indicators
    # ----------------------------

    ma5 = safe_float(row["MA5"])
    ma10 = safe_float(row["MA10"])
    ma20 = safe_float(row["MA20"])
    ma60 = safe_float(row["MA60"])
    ma120 = safe_float(row["MA120"])
    ma240 = safe_float(row["MA240"])

    rsi = safe_float(row["RSI14"])
    atr = safe_float(row["ATR14"])

    high20 = safe_float(row["HIGH20"])

    low10 = safe_float(row["LOW10"])
    low20 = safe_float(row["LOW20"])
    low60 = safe_float(row["LOW60"])

    ret20 = safe_float(row["RET20"])
    ret60 = safe_float(row["RET60"])

    trend_score, trend_reasons = calculate_trend_score(row)

    breakout_score, breakout_reasons, breakout_distance = (
        calculate_breakout_score(row)
    )

    volume_score, volume_ratio, volume_reasons = (
        calculate_volume_score(row)
    )

    rs_score, rs20, rs60, rs_reasons = (
        calculate_rs_score(row, market_ref)
    )

    distance_ma20 = (
        pct(price, ma20)
        if ma20 else 0
    )

    stock = {
        "code": code,
        "name": info["name"],
        "market": market,
        "industry": info["industry"],

        "ticker": ticker,
        "groups": groups,
        "is_etf": is_etf,

        "date": df.index[-1].strftime("%Y-%m-%d"),

        "price": r2(price),

        "ma5": r2(ma5),
        "ma10": r2(ma10),
        "ma20": r2(ma20),
        "ma60": r2(ma60),
        "ma120": r2(ma120),
        "ma240": r2(ma240),

        "rsi": r2(rsi),
        "atr14": r2(atr),

        "high20": r2(high20),

        "low10": r2(low10),
        "low20": r2(low20),
        "low60": r2(low60),

        "ret20": r2(ret20),
        "ret60": r2(ret60),

        "volume_ratio": r2(volume_ratio),

        "distance_ma20_pct": r2(distance_ma20),
        "breakout_distance_pct": r2(breakout_distance),

        "trend_score": trend_score,
        "breakout_score": breakout_score,
        "volume_score": volume_score,
        "rs_score": rs_score,

        "rs20": r2(rs20),
        "rs60": r2(rs60),

        "trend_reasons": trend_reasons,
        "breakout_reasons": breakout_reasons,
        "volume_reasons": volume_reasons,
        "rs_reasons": rs_reasons,

        "chart": build_chart_data(df),

        # V6.1 可接新聞
        "news": [],
    }

    return stock


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(market):
    if not market:
        return {
            "status": "Neutral",
            "label": "🟡 Neutral",
        }

    price = safe_float(market.get("price"))
    ma20 = safe_float(market.get("ma20"))
    ma60 = safe_float(market.get("ma60"))
    ret20 = safe_float(market.get("ret20"))

    if price > ma20 > ma60 and ret20 > 0:
        return {
            "status": "Risk-On",
            "label": "🟢 Risk-On",
        }

    if price < ma20 and ret20 < -3:
        return {
            "status": "Risk-Off",
            "label": "🔴 Risk-Off",
        }

    return {
        "status": "Neutral",
        "label": "🟡 Neutral",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Taiwan Stock Radar V6")
    print("=" * 60)

    codes = get_stock_codes()

    print(f"Universe: {len(codes)}")

    market_ref = get_market_reference()

    print("Market:", market_ref)

    raw_stocks = []

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    for i, code in enumerate(codes, 1):

        print(f"[{i}/{len(codes)}] {code}")

        try:
            stock = scan_one(code, market_ref)

            if stock:
                raw_stocks.append(stock)

        except Exception as e:
            print(f"[ERROR] {code}: {e}")

    print(f"Valid stocks: {len(raw_stocks)}")

    # --------------------------------------------------------
    # Sector strength
    # --------------------------------------------------------

    sector_map = build_sector_strength(raw_stocks)

    # --------------------------------------------------------
    # Second pass
    # --------------------------------------------------------

    for stock in raw_stocks:

        sector_score, best_sector, sector_detail = (
            apply_sector_score(stock, sector_map)
        )

        stock["sector_score"] = sector_score
        stock["best_sector"] = best_sector
        stock["sector_detail"] = sector_detail

        # Scores
        stock["next_day_score"] = calculate_next_day_score(stock)
        stock["ready_score"] = calculate_ready_score(stock)
        stock["mid_long_score"] = calculate_mid_long_score(stock)

        # Signals
        stock["next_day_signal"] = classify_next_day(stock)
        stock["ready_signal"] = classify_ready(stock)
        stock["mid_long_signal"] = classify_mid_long(stock)

        # Trade plan
        stock["trade_plan"] = calculate_trade_plan(stock)

        # Reasons
        reasons, risks = build_reasons(stock)

        stock["recommendation_reasons"] = reasons
        stock["risk_reasons"] = risks

        stock["next_day_action"] = build_next_day_action(stock)

        # Three-MA flag
        stock["above_3ma"] = (
            stock["price"] > stock["ma5"]
            and stock["price"] > stock["ma10"]
            and stock["price"] > stock["ma20"]
        )

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    next_day = sorted(
        raw_stocks,
        key=lambda x: (
            x["next_day_score"],
            x["volume_ratio"],
            x["rs20"],
        ),
        reverse=True,
    )

    ready = sorted(
        raw_stocks,
        key=lambda x: (
            x["ready_score"],
            x["rs20"],
        ),
        reverse=True,
    )

    mid_long = sorted(
        raw_stocks,
        key=lambda x: (
            x["mid_long_score"],
            x["rs60"],
        ),
        reverse=True,
    )

    radar = sorted(
        raw_stocks,
        key=lambda x: (
            max(
                x["next_day_score"],
                x["ready_score"],
                x["mid_long_score"],
            ),
            x["next_day_score"],
        ),
        reverse=True,
    )

    # Ranking fields
    for rank, stock in enumerate(next_day, 1):
        stock["next_day_rank"] = rank

    for rank, stock in enumerate(ready, 1):
        stock["ready_rank"] = rank

    for rank, stock in enumerate(mid_long, 1):
        stock["mid_long_rank"] = rank

    # --------------------------------------------------------
    # Top lists
    # --------------------------------------------------------

    next_day_top = [
        x for x in next_day
        if x["next_day_signal"] == "明日進場候選"
    ][:NEXT_DAY_TOP]

    # 如果不足10檔，補最高分等待確認
    if len(next_day_top) < NEXT_DAY_TOP:
        used = {x["code"] for x in next_day_top}

        extras = [
            x for x in next_day
            if x["code"] not in used
            and x["next_day_signal"] != "過熱／不追價"
        ]

        next_day_top += extras[
            : NEXT_DAY_TOP - len(next_day_top)
        ]

    ready_top = [
        x for x in ready
        if x["ready_signal"] in [
            "準備進場",
            "持續觀察",
        ]
    ][:READY_TOP]

    mid_long_top = [
        x for x in mid_long
        if x["mid_long_signal"] in [
            "中長期趨勢強",
            "中長期持續追蹤",
        ]
    ][:MID_LONG_TOP]

    radar_top = radar[:RADAR_TOP]

    # --------------------------------------------------------
    # Sector ranking
    # --------------------------------------------------------

    sector_ranking = sorted(
        sector_map.values(),
        key=lambda x: x["rank"],
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    market_regime = calculate_market_regime(market_ref)

    # --------------------------------------------------------
    # Signal counts
    # --------------------------------------------------------

    signal_counts = {}

    for stock in raw_stocks:
        sig = stock["next_day_signal"]
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    now = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y-%m-%d %H:%M")

    payload = {
        "version": VERSION,
        "updated": now,
        "timezone": TIMEZONE,

        "strategy": (
            "V6 Entry & Position Radar: "
            "Trend + Breakout + Volume + Relative Strength "
            "+ Sector Strength + Overheat Control"
        ),

        "universe_count": len(codes),
        "valid_count": len(raw_stocks),

        "market_regime": market_regime,
        "market_reference": {
            k: r2(v)
            for k, v in market_ref.items()
        },

        "signal_counts": signal_counts,

        "sector_ranking": sector_ranking,

        # 首頁雷達
        "next_day_top": next_day_top,
        "ready_top": ready_top,
        "mid_long_top": mid_long_top,

        # Top 80
        "stocks": radar_top,

        # Full Universe:
        # 讓國巨等即使沒進 Top80，也能搜尋得到
        "all_stocks": raw_stocks,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    os.makedirs("docs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with open(
        "docs/data.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_rows = []

    for stock in raw_stocks:

        row = {
            k: v
            for k, v in stock.items()
            if k not in [
                "chart",
                "news",
                "sector_detail",
                "trade_plan",
            ]
        }

        row["groups"] = ",".join(stock["groups"])
        row["recommendation_reasons"] = " | ".join(
            stock["recommendation_reasons"]
        )
        row["risk_reasons"] = " | ".join(
            stock["risk_reasons"]
        )

        for key, value in stock["trade_plan"].items():
            row[key] = value

        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        "data/signals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V6 COMPLETE")
    print("=" * 60)

    print("Updated:", now)
    print("Market:", market_regime["label"])
    print("Valid stocks:", len(raw_stocks))

    print()
    print("Tomorrow Top Picks:")

    for stock in next_day_top:
        plan = stock["trade_plan"]

        print(
            f"{stock['code']} "
            f"{stock['name']} "
            f"Score={stock['next_day_score']} "
            f"Entry={plan['entry_low']}-{plan['entry_high']} "
            f"Breakout={plan['breakout_price']} "
            f"Stop={plan['short_stop']}"
        )

    print()
    print("Top Sectors:")

    for sector in sector_ranking[:10]:
        print(
            f"#{sector['rank']} "
            f"{sector['group']} "
            f"Score={sector['score']}"
        )


if __name__ == "__main__":
    main()
