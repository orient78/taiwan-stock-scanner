import os
import json
import math
import time
import random
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

VERSION = "V6.2"
TIMEZONE = "Asia/Taipei"

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000
CHART_DAYS = 120

NEXT_DAY_TOP = 10
READY_TOP = 20
MID_LONG_TOP = 20
RADAR_TOP = 80

MAX_DOWNLOAD_RETRIES = 3
MIN_DOWNLOAD_SUCCESS_RATE = 0.50

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
# ä¸­æåç¨± / å¸å ´
# ============================================================

def get_stock_info(code):
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

    return {"name": code, "market": "", "industry": ""}


# ============================================================
# V6.2 TWSE-ONLY UNIVERSE
# ============================================================

def get_twse_codes():
    """
    Use twstock as the market master and keep TWSE-listed securities only.
    Ordinary listed stocks are included. The ETF codes already used by this
    project are also retained. OTC/TPEX securities are excluded completely.
    """
    codes = []

    for code, item in twstock.codes.items():
        try:
            market = str(getattr(item, "market", "") or "")
            security_type = str(getattr(item, "type", "") or "")

            if market != "ä¸å¸":
                continue

            # Taiwan common stock codes are four numeric digits.
            # Keep the project's listed ETF universe as well.
            is_common_stock = code.isdigit() and len(code) == 4
            is_project_etf = code in ETF_CODES

            if is_common_stock or is_project_etf:
                codes.append(code)
        except Exception:
            continue

    return sorted(set(codes))


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, adjust=False, min_periods=period
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

    for period in [5, 10, 20, 60, 120, 240]:
        df[f"MA{period}"] = df["Close"].rolling(period).mean()

    df["RSI14"] = calculate_rsi(df["Close"])
    df["VOL5"] = df["Volume"].rolling(5).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    df["HIGH20"] = df["High"].rolling(20).max().shift(1)
    df["HIGH60"] = df["High"].rolling(60).max().shift(1)

    df["ATR14"] = calculate_atr(df)

    df["LOW10"] = df["Low"].rolling(10).min().shift(1)
    df["LOW20"] = df["Low"].rolling(20).min().shift(1)
    df["LOW60"] = df["Low"].rolling(60).min().shift(1)

    df["RET5"] = df["Close"].pct_change(5) * 100
    df["RET20"] = df["Close"].pct_change(20) * 100
    df["RET60"] = df["Close"].pct_change(60) * 100
    df["RET120"] = df["Close"].pct_change(120) * 100

    return df


# ============================================================
# YFINANCE NORMALIZER
# ============================================================

def normalize_yfinance_df(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not set(required).issubset(df.columns):
        return None

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()

    return df


# ============================================================
# MARKET DATA
# ============================================================

def download_history(code, max_retries=MAX_DOWNLOAD_RETRIES):
    """Download TWSE data only. V6.2 never queries .TWO."""
    ticker = f"{code}.TW"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DOWNLOAD] {ticker} attempt {attempt}/{max_retries}")

            df = yf.download(
                ticker,
                period="18mo",
                interval="1d",
                auto_adjust=True,
                repair=False,
                progress=False,
                threads=False,
                timeout=20,
            )

            df = normalize_yfinance_df(df)

            if df is None or df.empty:
                raise RuntimeError("empty/invalid dataframe")

            if len(df) < 80:
                raise RuntimeError(f"insufficient history: {len(df)} days")

            last_close = safe_float(df["Close"].iloc[-1])
            if last_close <= 0:
                raise RuntimeError("invalid last close")

            print(f"[OK] {ticker}: {len(df)} days, close={last_close:.2f}")
            return ticker, "ä¸å¸", df

        except Exception as e:
            print(f"[WARN] {ticker} attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_seconds = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"[WAIT] {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    print(f"[FAIL] {ticker}: all retries failed")
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

def get_market_reference(max_retries=MAX_DOWNLOAD_RETRIES):
    ticker = "^TWII"

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[MARKET] Download {ticker} "
                f"attempt {attempt}/{max_retries}"
            )

            df = yf.download(
                ticker,
                period="18mo",
                interval="1d",
                auto_adjust=True,
                repair=False,
                progress=False,
                threads=False,
                timeout=20,
            )

            df = normalize_yfinance_df(df)

            if df is None or df.empty:
                raise RuntimeError("empty market dataframe")

            if len(df) < 60:
                raise RuntimeError(
                    f"insufficient market history: {len(df)}"
                )

            close = df["Close"].dropna()

            if len(close) < 60:
                raise RuntimeError("insufficient valid Close history")

            result = {
                "ret5": safe_float(close.pct_change(5).iloc[-1] * 100),
                "ret20": safe_float(close.pct_change(20).iloc[-1] * 100),
                "ret60": safe_float(close.pct_change(60).iloc[-1] * 100),
                "ma20": safe_float(close.rolling(20).mean().iloc[-1]),
                "ma60": safe_float(close.rolling(60).mean().iloc[-1]),
                "price": safe_float(close.iloc[-1]),
            }

            print("[MARKET OK]", result)
            return result

        except Exception as e:
            print(
                f"[MARKET WARN] attempt "
                f"{attempt}/{max_retries} failed: {e}"
            )

            if attempt < max_retries:
                wait_seconds = (
                    (2 ** attempt) + random.uniform(0.5, 1.5)
                )
                print(f"[MARKET WAIT] {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    print("[MARKET ERROR] All attempts failed")
    return {}


# ============================================================
# CORE V6 SCORES
# ============================================================

def calculate_trend_score(row):
    score = 0
    reasons = []

    price = safe_float(row["Close"])
    ma5 = safe_float(row["MA5"])
    ma10 = safe_float(row["MA10"])
    ma20 = safe_float(row["MA20"])
    ma60 = safe_float(row["MA60"])

    if price > ma20:
        score += 6
        reasons.append("è¡å¹ç«ä¸20æ¥åç·")

    if ma5 > ma10 > ma20:
        score += 8
        reasons.append("MA5 > MA10 > MA20ï¼å¤é ­æå")

    if ma20 > ma60:
        score += 6
        reasons.append("MA20ä½æ¼MA60ä¹ä¸")

    if ma20 > 0:
        score += 5
        reasons.append("ä¸­ç­æè¶¨å¢ç¶­æåå¤")

    return min(score, 25), reasons


def calculate_breakout_score(row):
    price = safe_float(row["Close"])
    high20 = safe_float(row["HIGH20"])

    if not high20:
        return 0, [], 999

    distance = (high20 - price) / high20 * 100
    score = 0
    reasons = []

    if price >= high20:
        score = 25
        reasons.append("å·²çªç ´20æ¥é«é»")
    elif distance <= 1:
        score = 23
        reasons.append("è·20æ¥çªç ´å¹ä¸å°1%")
    elif distance <= 2:
        score = 21
        reasons.append("éå¸¸æ¥è¿20æ¥çªç ´")
    elif distance <= 4:
        score = 17
        reasons.append("æ¥è¿20æ¥çªç ´å")
    elif distance <= 7:
        score = 10
        reasons.append("è·çªç ´ä½ç½®å°å¯")

    return score, reasons, distance


def calculate_volume_score(row):
    volume = safe_float(row["Volume"])
    vol20 = safe_float(row["VOL20"])
    ratio = volume / vol20 if vol20 else 0

    score = 0
    reasons = []

    if ratio >= 2:
        score = 20
        reasons.append("æäº¤éè¶é20æ¥åé2å")
    elif ratio >= 1.5:
        score = 18
        reasons.append("æäº¤éæé¡¯æ¾å¤§")
    elif ratio >= 1.2:
        score = 15
        reasons.append("éè½éå§æ´å¼µ")
    elif ratio >= 1:
        score = 10
        reasons.append("æäº¤éé«æ¼20æ¥åé")
    elif ratio >= 0.75:
        score = 5

    return score, ratio, reasons


def calculate_rs_score(row, market):
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
        reasons.append(f"è¿20æ¥è¡¨ç¾åªæ¼å¤§ç¤ {rs20:.1f}%")

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

    return {item["group"]: item for item in ordered}


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

    sector_component = round(best["score"] / 100 * 15)

    return sector_component, best["group"], best


# ============================================================
# ENTRY / STOP LOSS ENGINE
# ============================================================

def calculate_trade_plan(stock):
    price = stock["price"]
    ma10 = stock["ma10"]
    ma20 = stock["ma20"]
    ma60 = stock["ma60"]

    high20 = stock["high20"]
    atr = stock["atr14"]

    low10 = stock["low10"]
    low20 = stock["low20"]
    low60 = stock["low60"]

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

    entry_high = min(price, support * 1.015)

    if entry_low > entry_high:
        entry_low = price * 0.98
        entry_high = price

    breakout = high20 if high20 > 0 else price

    chase_by_pct = breakout * 1.03

    if atr:
        chase_by_atr = breakout + atr * 0.75
        chase_limit = min(chase_by_pct, chase_by_atr)
    else:
        chase_limit = chase_by_pct

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

    max_stop = entry_low * 0.985
    short_stop = min(short_stop, max_stop)

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
    score = (
        stock["trend_score"]
        + stock["breakout_score"]
        + stock["volume_score"]
        + stock["rs_score"]
        + stock["sector_score"]
    )

    rsi = stock["rsi"]

    if rsi >= 80:
        score -= 25
    elif rsi >= 75:
        score -= 15
    elif rsi >= 72:
        score -= 7

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
        return "éç±ï¼ä¸è¿½å¹"

    if (
        score >= 80
        and stock["breakout_distance_pct"] <= 2
        and stock["volume_ratio"] >= 1.2
    ):
        return "ææ¥é²å ´åé¸"

    if score >= 68:
        return "ç­å¾ææ¥ç¢ºèª"

    return "æ«ä¸èæ®"


def classify_ready(stock):
    score = stock["ready_score"]

    if stock["rsi"] >= 75:
        return "éç±"

    if score >= 80:
        return "æºåé²å ´"

    if score >= 65:
        return "æçºè§å¯"

    return "å°æªæç"


def classify_mid_long(stock):
    score = stock["mid_long_score"]

    if score >= 80:
        return "ä¸­é·æè¶¨å¢å¼·"

    if score >= 65:
        return "ä¸­é·ææçºè¿½è¹¤"

    return "ä¸­é·æä¸è¬"


# ============================================================
# REASONS
# ============================================================

def build_reasons(stock):
    reasons = []
    risks = []

    if stock["ma5"] > stock["ma10"] > stock["ma20"]:
        reasons.append("ç­æåç·åå¤é ­æå")

    if stock["ma20"] > stock["ma60"]:
        reasons.append("ä¸­æè¶¨å¢ç¶­æåä¸")

    if stock["breakout_distance_pct"] <= 2:
        reasons.append("å·²éå¸¸æ¥è¿20æ¥çªç ´ä½ç½®")
    elif stock["breakout_distance_pct"] <= 5:
        reasons.append("æ­£å¨æ¥è¿20æ¥å£åå")

    if stock["volume_ratio"] >= 1.5:
        reasons.append(
            f"æäº¤éæ¾å¤§è³20æ¥åé {stock['volume_ratio']:.2f} å"
        )
    elif stock["volume_ratio"] >= 1.2:
        reasons.append("æäº¤ééå§æ´å¼µ")

    if stock["rs20"] > 0:
        reasons.append(
            f"è¿20æ¥ç¸å°å¤§ç¤å¼· {stock['rs20']:.1f}%"
        )

    if stock["best_sector"]:
        reasons.append(
            f"{stock['best_sector']}æç¾¤ç¸å°å¼·å¢"
        )

    if stock["rsi"] >= 75:
        risks.append("RSIé²å¥éç±åï¼ä¸é©åè¿½å¹")
    elif stock["rsi"] >= 70:
        risks.append("RSIåé«ï¼æ³¨æéæ¥è¿½å¹é¢¨éª")

    if stock["distance_ma20_pct"] >= 12:
        risks.append("è¡å¹èMA20ä¹é¢éå¤§")

    if stock["volume_ratio"] < 0.8:
        risks.append("ç®åéè½ä»ä¸è¶³")

    if stock["breakout_distance_pct"] > 7:
        risks.append("è·é¢çªç ´ä½ç½®ä»è¼é ")

    if not reasons:
        reasons.append("ç®åä»¥æè¡çµæ§è§å¯çºä¸»")

    if not risks:
        risks.append("ä»éè§å¯éæ¥éç¤èæäº¤éç¢ºèª")

    return reasons[:5], risks[:4]


# ============================================================
# NEXT DAY ACTION
# ============================================================

def build_next_day_action(stock):
    p = stock["trade_plan"]

    breakout = p["breakout_price"]
    chase = p["chase_limit"]

    return (
        f"ææ¥è¥çªç ´ {breakout:.2f} ä¸éè½åæ­¥æ¾å¤§ï¼"
        f"å¯è¦çºé²å ´è§¸ç¼ï¼"
        f"è¥ç´æ¥è·³ç©ºé«æ¼ {chase:.2f}ï¼ä¸å»ºè­°è¿½å¹ã"
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

    if not is_etf:
        if price < MIN_PRICE:
            return None

        if avg_value < MIN_AVG_DAILY_VALUE:
            return None

    info = get_stock_info(code)
    market = info["market"] or detected_market
    groups = get_groups(code)
    if not groups and info.get("industry"):
        groups = [info["industry"]]

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

    distance_ma20 = pct(price, ma20) if ma20 else 0

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
            "label": "ð¡ Neutral",
        }

    price = safe_float(market.get("price"))
    ma20 = safe_float(market.get("ma20"))
    ma60 = safe_float(market.get("ma60"))
    ret20 = safe_float(market.get("ret20"))

    if price > ma20 > ma60 and ret20 > 0:
        return {
            "status": "Risk-On",
            "label": "ð¢ Risk-On",
        }

    if price < ma20 and ret20 < -3:
        return {
            "status": "Risk-Off",
            "label": "ð´ Risk-Off",
        }

    return {
        "status": "Neutral",
        "label": "ð¡ Neutral",
    }


# ============================================================
# DATA SAFETY
# ============================================================

def validate_scan_result(universe_count, valid_count, market_ref):
    print()
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    print(f"Universe : {universe_count}")
    print(f"Valid    : {valid_count}")

    if universe_count <= 0:
        raise RuntimeError(
            "CRITICAL: Stock universe is empty. Abort publication."
        )

    if valid_count == 0:
        raise RuntimeError(
            "CRITICAL: 0 valid stocks. Market data download likely "
            "failed. Existing data.json will NOT be overwritten."
        )

    success_rate = valid_count / universe_count

    print(f"Success  : {success_rate:.1%}")

    if success_rate < MIN_DOWNLOAD_SUCCESS_RATE:
        raise RuntimeError(
            f"CRITICAL: Only {valid_count}/{universe_count} "
            f"({success_rate:.1%}) stocks survived scan. "
            "Abort publication to protect previous data."
        )

    if not market_ref:
        raise RuntimeError(
            "CRITICAL: ^TWII market reference unavailable. "
            "Abort publication."
        )

    print("[VALIDATION OK] Scan data is healthy.")
    return success_rate


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Taiwan Stock Radar V6.2 - TWSE Only")
    print("=" * 60)

    codes = get_twse_codes()
    print(f"Universe: {len(codes)}")

    market_ref = get_market_reference()
    print("Market:", market_ref)

    # Fail early. This prevents a Yahoo outage from creating an empty release.
    if not market_ref:
        raise RuntimeError(
            "CRITICAL: Unable to download ^TWII market data. "
            "Abort before scanning stocks."
        )

    raw_stocks = []

    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}")

        try:
            stock = scan_one(code, market_ref)

            if stock:
                raw_stocks.append(stock)

        except Exception as e:
            print(f"[ERROR] {code}: {e}")

    print(f"Valid stocks: {len(raw_stocks)}")

    # IMPORTANT: validate before creating/writing any output file.
    validate_scan_result(
        universe_count=len(codes),
        valid_count=len(raw_stocks),
        market_ref=market_ref,
    )

    sector_map = build_sector_strength(raw_stocks)

    for stock in raw_stocks:
        sector_score, best_sector, sector_detail = (
            apply_sector_score(stock, sector_map)
        )

        stock["sector_score"] = sector_score
        stock["best_sector"] = best_sector
        stock["sector_detail"] = sector_detail

        stock["next_day_score"] = calculate_next_day_score(stock)
        stock["ready_score"] = calculate_ready_score(stock)
        stock["mid_long_score"] = calculate_mid_long_score(stock)

        stock["next_day_signal"] = classify_next_day(stock)
        stock["ready_signal"] = classify_ready(stock)
        stock["mid_long_signal"] = classify_mid_long(stock)

        stock["trade_plan"] = calculate_trade_plan(stock)

        reasons, risks = build_reasons(stock)

        stock["recommendation_reasons"] = reasons
        stock["risk_reasons"] = risks
        stock["next_day_action"] = build_next_day_action(stock)

        stock["above_3ma"] = (
            stock["price"] > stock["ma5"]
            and stock["price"] > stock["ma10"]
            and stock["price"] > stock["ma20"]
        )

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

    for rank, stock in enumerate(next_day, 1):
        stock["next_day_rank"] = rank

    for rank, stock in enumerate(ready, 1):
        stock["ready_rank"] = rank

    for rank, stock in enumerate(mid_long, 1):
        stock["mid_long_rank"] = rank

    next_day_top = [
        x for x in next_day
        if x["next_day_signal"] == "ææ¥é²å ´åé¸"
    ][:NEXT_DAY_TOP]

    # V6.1: do not pad strict next-day entries just to reach 10 names.
    # Keep a separate watchlist for the strongest non-overheated candidates.
    used = {x["code"] for x in next_day_top}
    next_day_watch = [
        x for x in next_day
        if x["code"] not in used
        and x["next_day_signal"] != "éç±ï¼ä¸è¿½å¹"
    ][:NEXT_DAY_TOP]

    ready_top = [
        x for x in ready
        if x["ready_signal"] in [
            "æºåé²å ´",
            "æçºè§å¯",
        ]
    ][:READY_TOP]

    mid_long_top = [
        x for x in mid_long
        if x["mid_long_signal"] in [
            "ä¸­é·æè¶¨å¢å¼·",
            "ä¸­é·ææçºè¿½è¹¤",
        ]
    ][:MID_LONG_TOP]

    radar_top = radar[:RADAR_TOP]

    sector_ranking = sorted(
        sector_map.values(),
        key=lambda x: x["rank"],
    )

    market_regime = calculate_market_regime(market_ref)

    signal_counts = {}

    for stock in raw_stocks:
        sig = stock["next_day_signal"]
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    now = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y-%m-%d %H:%M")

    payload = {
        "version": VERSION,
        "updated": now,
        "timezone": TIMEZONE,
        "strategy": (
            "V6.2 TWSE Entry & Position Radar: "
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
        "next_day_top": next_day_top,
        "next_day_watch": next_day_watch,
        "ready_top": ready_top,
        "mid_long_top": mid_long_top,
        "stocks": radar_top,
        "all_stocks": raw_stocks,
    }

    # Only write after successful validation.
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

    print("[SAVE OK] docs/data.json")

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

    print("[SAVE OK] data/signals.csv")

    print()
    print("=" * 60)
    print("V6.2 COMPLETE")
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
