import os
import json
import math
import time
import random
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import twstock

from stock_pool import get_groups


# ============================================================
# V6.5 CONFIG
# ============================================================

VERSION = "V6.6"
TIMEZONE = "Asia/Taipei"

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000
CHART_DAYS = 120

READY_TOP = 20
MID_LONG_TOP = 20
RADAR_TOP = 80

MAX_DOWNLOAD_RETRIES = 3
MIN_DOWNLOAD_SUCCESS_RATE = 0.85

# Sector statistics: avoid tiny groups receiving extreme scores.
MIN_SECTOR_MEMBERS = 5

# Tomorrow-entry risk controls.
MIN_SHORT_RISK_PCT = 1.5
MAX_SHORT_RISK_PCT = 7.0
MIN_REAL_RR = 1.35
MIN_ENTRY_QUALITY = 60


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
# Full TWSE-listed + TPEx-OTC common-stock universe (ETF/ETN excluded).
# ============================================================

TWSE_CODES = set()
TPEX_CODES = set()

def get_twse_codes():
    global TWSE_CODES, TPEX_CODES
    twse = getattr(twstock, "twse", {}) or {}
    tpex = getattr(twstock, "tpex", {}) or {}
    if not twse or not tpex:
        raise RuntimeError("CRITICAL: TWSE or TPEx symbol list is unavailable.")

    def common_codes(exchange_codes):
        result = set()
        for raw_code in exchange_codes:
            code = str(raw_code).strip()
            if code.isdigit() and len(code) == 4 and not code.startswith("0"):
                if code in twstock.codes:
                    result.add(code)
        return result

    TWSE_CODES = common_codes(twse)
    TPEX_CODES = common_codes(tpex)
    if len(TWSE_CODES) < 500 or len(TPEX_CODES) < 300:
        raise RuntimeError(
            f"CRITICAL: suspicious market universe sizes "
            f"(TWSE={len(TWSE_CODES)}, TPEx={len(TPEX_CODES)})."
        )
    codes = sorted(TWSE_CODES | TPEX_CODES)
    if "2330" not in TWSE_CODES:
        raise RuntimeError("CRITICAL: 2330 is missing from TWSE universe.")
    if "5483" not in TPEX_CODES:
        raise RuntimeError("CRITICAL: TPEx symbol 5483 (中美晶) is missing.")
    print(f"[UNIVERSE] TWSE={len(TWSE_CODES)}, TPEx={len(TPEX_CODES)}, total={len(codes)}")
    print("[CHECK] 2330 on TWSE: True")
    print("[CHECK] 5483 on TPEx: True")
    print("[CHECK] Yahoo suffixes: .TW and .TWO")
    return codes



# ============================================================
# TWSE T86 法人資料
# ============================================================

def _t86_int(value):
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0


def fetch_t86(date_string):
    query = urllib.parse.urlencode(
        {
            "response": "json",
            "date": date_string,
            "selectType": "ALLBUT0999",
        }
    )
    url = "https://www.twse.com.tw/rwd/zh/fund/T86?" + query
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8-sig"))

    fields = payload.get("fields") or []
    rows = payload.get("data") or []
    if not fields or not rows:
        return {}

    field_index = {name: index for index, name in enumerate(fields)}
    source_fields = {
        "foreign_net_shares": "外陸資買賣超股數(不含外資自營商)",
        "trust_net_shares": "投信買賣超股數",
        "dealer_net_shares": "自營商買賣超股數",
        "institutional_net_shares": "三大法人買賣超股數",
    }
    required = ["證券代號", *source_fields.values()]
    if any(name not in field_index for name in required):
        raise RuntimeError("TWSE T86 schema changed")

    result = {}
    for row in rows:
        code = str(row[field_index["證券代號"]]).strip()
        if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
            continue
        result[code] = {
            key: _t86_int(row[field_index[field]])
            for key, field in source_fields.items()
        }

    return result


def get_t86_history(days=5):
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    history = []

    for offset in range(14):
        date = today - timedelta(days=offset)
        if date.weekday() >= 5:
            continue

        try:
            daily = fetch_t86(date.strftime("%Y%m%d"))
            if daily:
                history.append((date.strftime("%Y-%m-%d"), daily))
                if len(history) >= days:
                    break
        except Exception as error:
            print("[T86 WARN]", date, error)

    if not history:
        return {}, {"status": "unavailable", "date": "", "days": 0}

    latest_date, latest_data = history[0]
    result = {}

    for code, latest in latest_data.items():
        stock_data = dict(latest)
        for key, value in latest.items():
            stock_data[key.replace("_shares", "_lots")] = r2(value / 1000)

        streak_types = {
            "foreign": "foreign_net_shares",
            "trust": "trust_net_shares",
            "institutional": "institutional_net_shares",
        }
        for name, key in streak_types.items():
            for direction, positive in (("buy", True), ("sell", False)):
                streak = 0
                for _, daily_data in history:
                    if code not in daily_data:
                        break
                    value = daily_data[code][key]
                    matches = value > 0 if positive else value < 0
                    if not matches:
                        break
                    streak += 1
                stock_data[f"{name}_{direction}_streak"] = streak

        result[code] = stock_data

    return result, {
        "status": "ok",
        "date": latest_date,
        "days": len(history),
    }

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

    # Previous highs/lows only: avoids using today's high/low as the breakout reference.
    df["HIGH20"] = df["High"].rolling(20).max().shift(1)
    df["HIGH60"] = df["High"].rolling(60).max().shift(1)
    df["HIGH120"] = df["High"].rolling(120).max().shift(1)

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
    ticker = f"{code}.TWO" if code in TPEX_CODES else f"{code}.TW"
    detected_market = "上櫃" if code in TPEX_CODES else "上市"

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
            return ticker, detected_market, df, None

        except Exception as e:
            print(f"[WARN] {ticker} attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_seconds = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"[WAIT] {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    print(f"[FAIL] {ticker}: all retries failed")
    return None, None, None, "download_failed"


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
            print(f"[MARKET] Download {ticker} attempt {attempt}/{max_retries}")

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
                raise RuntimeError(f"insufficient market history: {len(df)}")

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
            print(f"[MARKET WARN] attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                wait_seconds = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"[MARKET WAIT] {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    print("[MARKET ERROR] All attempts failed")
    return {}


# ============================================================
# CORE SCORES
# ============================================================

def calculate_trend_score(row):
    score, reasons = 0, []

    price = safe_float(row["Close"])
    ma5 = safe_float(row["MA5"])
    ma10 = safe_float(row["MA10"])
    ma20 = safe_float(row["MA20"])
    ma60 = safe_float(row["MA60"])

    if ma20 > 0 and price > ma20:
        score += 6
        reasons.append("股價站上20日均線")

    if ma5 > 0 and ma10 > 0 and ma20 > 0 and ma5 > ma10 > ma20:
        score += 8
        reasons.append("MA5 > MA10 > MA20，多頭排列")

    if ma20 > 0 and ma60 > 0 and ma20 > ma60:
        score += 6
        reasons.append("MA20位於MA60之上")

    # V6.4 FIX: no unconditional +5 merely because MA20 exists.
    if (
        price > ma20 > ma60 > 0
        and ma5 > ma10 > ma20
    ):
        score += 5
        reasons.append("短中期趨勢結構完整偏多")

    return min(score, 25), reasons


def calculate_breakout_score(row):
    price = safe_float(row["Close"])
    high20 = safe_float(row["HIGH20"])

    if high20 <= 0:
        return 0, [], 999, 0

    distance = (high20 - price) / high20 * 100
    extension = max(0.0, (price / high20 - 1.0) * 100)

    score = 0
    reasons = []

    # V6.4 FIX: a stock far above the breakout level no longer gets 25/25.
    if price >= high20:
        if extension <= 1.5:
            score = 25
            reasons.append("剛突破20日高點，延伸幅度仍小")
        elif extension <= 3:
            score = 22
            reasons.append("已突破20日高點，仍在可控延伸區")
        elif extension <= 5:
            score = 16
            reasons.append("已突破20日高點，但短線已有延伸")
        elif extension <= 8:
            score = 8
            reasons.append("突破後漲幅偏大，追價風險提高")
        else:
            score = 0
            reasons.append("突破後延伸過大，不給突破追價分")
    elif 0 < distance <= 1:
        score = 23
        reasons.append("距20日突破價不到1%")
    elif 1 < distance <= 2:
        score = 21
        reasons.append("非常接近20日突破")
    elif 2 < distance <= 4:
        score = 17
        reasons.append("接近20日突破區")
    elif 4 < distance <= 7:
        score = 10
        reasons.append("距突破位置尚可")

    return score, reasons, distance, extension


def calculate_volume_score(row):
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
    rs20 = safe_float(row["RET20"]) - safe_float(market.get("ret20"))
    rs60 = safe_float(row["RET60"]) - safe_float(market.get("ret60"))

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

    raw_results = []

    for group, members in groups.items():
        if not members:
            continue

        rs_values = [x["rs20"] for x in members]
        trend_values = [
            1 if x["price"] > x["ma20"] > 0 else 0
            for x in members
        ]
        ret20_values = [x["ret20"] for x in members]

        avg_rs = safe_float(np.mean(rs_values))
        avg_ret20 = safe_float(np.mean(ret20_values))
        trend_ratio = safe_float(np.mean(trend_values))

        raw = (
            avg_rs * 0.45
            + avg_ret20 * 0.25
            + trend_ratio * 20 * 0.30
        )

        # V6.4 FIX: shrink tiny groups toward neutral instead of allowing
        # one or two stocks to create a "top sector".
        reliability = min(1.0, len(members) / MIN_SECTOR_MEMBERS)
        adjusted_raw = raw * reliability

        raw_results.append(
            {
                "group": group,
                "raw_score": safe_float(raw),
                "adjusted_raw_score": safe_float(adjusted_raw),
                "avg_rs20": r2(avg_rs),
                "avg_ret20": r2(avg_ret20),
                "trend_ratio": r2(trend_ratio * 100),
                "count": len(members),
                "reliability": r2(reliability),
            }
        )

    ordered = sorted(
        raw_results,
        key=lambda x: x["adjusted_raw_score"],
        reverse=True,
    )

    for rank, item in enumerate(ordered, 1):
        if len(ordered) <= 1:
            percentile_score = 100
        else:
            percentile = 1 - ((rank - 1) / (len(ordered) - 1))
            percentile_score = round(percentile * 100)

        # Tiny sectors cannot receive a full-strength sector score.
        item["score"] = round(percentile_score * item["reliability"])

    # V6.4.1 FIX: displayed rank follows displayed final score.
    ordered = sorted(
        ordered,
        key=lambda x: (x["score"], x["adjusted_raw_score"]),
        reverse=True,
    )
    for rank, item in enumerate(ordered, 1):
        item["rank"] = rank

    return {item["group"]: item for item in ordered}


def calculate_sector_detail_ex_self(stock, group, members):
    """
    V6.4: calculate sector contribution excluding the current stock.
    This prevents a strong stock from making its own sector strong and
    then receiving that strength back as an extra score.
    """
    peers = [x for x in members if x["code"] != stock["code"]]

    if not peers:
        return None

    rs_values = [x["rs20"] for x in peers]
    ret20_values = [x["ret20"] for x in peers]
    trend_values = [
        1 if x["price"] > x["ma20"] > 0 else 0
        for x in peers
    ]

    avg_rs = safe_float(np.mean(rs_values))
    avg_ret20 = safe_float(np.mean(ret20_values))
    trend_ratio = safe_float(np.mean(trend_values))

    raw = (
        avg_rs * 0.45
        + avg_ret20 * 0.25
        + trend_ratio * 20 * 0.30
    )

    reliability = min(1.0, len(peers) / MIN_SECTOR_MEMBERS)

    return {
        "group": group,
        "raw_score": raw,
        "adjusted_raw_score": raw * reliability,
        "peer_count": len(peers),
        "reliability": reliability,
    }


def apply_sector_score(stock, sector_map, group_members):
    candidates = []

    for group in stock["groups"]:
        published = sector_map.get(group)
        members = group_members.get(group, [])

        if not published or not members:
            continue

        ex_self = calculate_sector_detail_ex_self(stock, group, members)

        if not ex_self:
            continue

        # Use the published percentile as a base, but attenuate it with
        # ex-self reliability. This preserves V6 output scale (0..15)
        # while preventing self-contribution and tiny-group inflation.
        effective_score = published["score"] * ex_self["reliability"]

        candidates.append(
            (effective_score, group, published, ex_self)
        )

    if not candidates:
        return 0, "", None

    candidates.sort(key=lambda x: x[0], reverse=True)
    effective_score, group, published, ex_self = candidates[0]

    sector_component = round(effective_score / 100 * 15)

    detail = dict(published)
    detail["ex_self"] = {
        "peer_count": ex_self["peer_count"],
        "reliability": r2(ex_self["reliability"]),
        "adjusted_raw_score": r2(ex_self["adjusted_raw_score"]),
    }

    return sector_component, group, detail


# ============================================================
# ENTRY / STOP LOSS / REAL RR ENGINE
# ============================================================

def calculate_trade_plan(stock):
    price = stock["price"]
    ma10 = stock["ma10"]
    ma20 = stock["ma20"]
    ma60 = stock["ma60"]

    high20 = stock["high20"]
    high60 = stock["high60"]
    high120 = stock["high120"]
    atr = stock["atr14"]

    low10 = stock["low10"]
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

    # Ensure stop is not unrealistically close to the entry zone.
    closest_allowed_stop = entry_low * (1 - MIN_SHORT_RISK_PCT / 100)
    short_stop = min(short_stop, closest_allowed_stop)

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

    short_risk_pct = (
        risk / entry_high * 100
        if entry_high > 0 and risk > 0 else 0
    )

    long_risk_pct = (
        (price - long_stop) / price * 100
        if price else 0
    )

    # Find a real technical resistance above the proposed entry.
    resistances = sorted({
        r2(x)
        for x in [high60, high120]
        if x > entry_high * 1.005
    })

    resistance = resistances[0] if resistances else 0.0

    if risk > 0 and resistance > entry_high:
        real_reward = resistance - entry_high
        real_rr = real_reward / risk
        target1 = resistance
        target2 = max(resistance, entry_high + risk * 2)
        target_source = "technical_resistance"
    elif risk > 0:
        # No visible historical resistance above entry. Keep projected
        # targets for display, but mark RR as estimated rather than "real".
        target1 = entry_high + risk * 2
        target2 = entry_high + risk * 3
        real_rr = 2.0
        target_source = "projected_2R"
    else:
        target1 = price * 1.08
        target2 = price * 1.12
        real_rr = 0.0
        target_source = "fallback"

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
        "target1_return_pct": r2(pct(target1, price)),
        "target2_return_pct": r2(pct(target2, price)),
        "previous_high": r2(high20),
        "previous_low": r2(stock.get("low20",0)),
        "support": r2(support),
        "major_support": r2(
            max(
                [
                    value
                    for value in [ma60, stock.get("low20", 0), low60]
                    if 0 < value <= price
                ]
                or [price * 0.90]
            )
        ),
        "distance_support_pct": r2(pct(price,support)) if support else 0,
        "distance_previous_high_pct": r2(pct(price,high20)) if high20 else 0,
        # Keep old field for index.html compatibility.
        "risk_reward": r2(real_rr),
        "real_risk_reward": r2(real_rr),
        "target_source": target_source,
        "resistance_price": r2(resistance),
    }


# ============================================================
# NEXT DAY / ENTRY QUALITY SCORE
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

    extension = stock["breakout_extension_pct"]

    if extension > 8:
        score -= 15
    elif extension > 5:
        score -= 8
    elif extension > 3:
        score -= 3

    return max(0, min(round(score), 100))


def calculate_entry_quality_score(stock):
    """
    Separate "strong stock" from "good next-session entry".
    """
    score = 0

    d = stock["breakout_distance_pct"]
    extension = stock["breakout_extension_pct"]
    volume_ratio = stock["volume_ratio"]
    rsi = stock["rsi"]
    ma20_dist = stock["distance_ma20_pct"]
    plan = stock["trade_plan"]
    risk_pct = plan["short_risk_pct"]
    rr = plan["real_risk_reward"]

    # Location vs breakout: 30
    if 0 <= d <= 1.5:
        score += 30
    elif -1.5 <= d < 0:
        score += 30
    elif 1.5 < d <= 3:
        score += 22
    elif -3 <= d < -1.5:
        score += 22
    elif 3 < d <= 5:
        score += 12

    # Volume confirmation: 20
    if 1.2 <= volume_ratio <= 2.5:
        score += 20
    elif 1.0 <= volume_ratio < 1.2:
        score += 12
    elif volume_ratio > 2.5:
        score += 12
    elif 0.8 <= volume_ratio < 1.0:
        score += 6

    # RSI quality: 15
    if 50 <= rsi < 70:
        score += 15
    elif 45 <= rsi < 72:
        score += 10
    elif rsi < 75:
        score += 5

    # MA20 extension quality: 15
    if 0 <= ma20_dist <= 6:
        score += 15
    elif 6 < ma20_dist <= 9:
        score += 8
    elif -3 <= ma20_dist < 0:
        score += 5

    # Stop risk: 10
    if MIN_SHORT_RISK_PCT <= risk_pct <= 5:
        score += 10
    elif 5 < risk_pct <= MAX_SHORT_RISK_PCT:
        score += 5

    # Reward/risk: 10
    if rr >= 2:
        score += 10
    elif rr >= MIN_REAL_RR:
        score += 6

    # Explicit penalty for excessive breakout extension.
    if extension > 8:
        score -= 20
    elif extension > 5:
        score -= 10

    return max(0, min(round(score), 100))


# ============================================================
# READY SCORE
# ============================================================

def calculate_ready_score(stock):
    score = 0

    if stock["price"] > stock["ma20"] > 0:
        score += 20

    if stock["ma5"] > stock["ma10"] > stock["ma20"] > 0:
        score += 20

    d = stock["breakout_distance_pct"]

    # V6.4 FIX: negative d (already broken out) must not fall into <=2.
    if 0 < d <= 1:
        score += 25
    elif 1 < d <= 2:
        score += 22
    elif 2 < d <= 4:
        score += 18
    elif 4 < d <= 7:
        score += 10
    elif -1.5 <= d <= 0:
        score += 18
    elif -3 <= d < -1.5:
        score += 10

    if 0.7 <= stock["volume_ratio"] <= 1.5:
        score += 10

    if stock["rs20"] > 0:
        score += 15

    if stock["sector_score"] >= 10:
        score += 10

    if stock["rsi"] >= 75:
        score -= 15

    if stock["breakout_extension_pct"] > 5:
        score -= 15

    return max(0, min(round(score), 100))


# ============================================================
# MID/LONG SCORE
# ============================================================

def calculate_mid_long_score(stock):
    score = 0
    price = stock["price"]

    if price > stock["ma20"] > 0:
        score += 10

    if stock["ma20"] > stock["ma60"] > 0:
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

def get_next_day_failed_gates(stock):
    """Return actionable tomorrow-entry gates that are not satisfied."""
    plan = stock["trade_plan"]
    failed = []

    if stock["rsi"] >= 75:
        failed.append("RSI>=75")
    if stock["distance_ma20_pct"] >= 12:
        failed.append("MA20乖離>=12%")
    if stock["breakout_extension_pct"] > 8:
        failed.append("突破延伸>8%")
    if stock["next_day_score"] < 80:
        failed.append("Score<80")
    if stock["entry_quality_score"] < MIN_ENTRY_QUALITY:
        failed.append(f"EntryQ<{MIN_ENTRY_QUALITY}")
    if not (-3 <= stock["breakout_distance_pct"] <= 2):
        failed.append("突破位置不佳")
    if stock["volume_ratio"] < 1.2:
        failed.append("量比<1.2")
    if stock["rs20"] <= 0:
        failed.append("RS20<=0")
    if not (
        MIN_SHORT_RISK_PCT
        <= plan["short_risk_pct"]
        <= MAX_SHORT_RISK_PCT
    ):
        failed.append(
            f"StopRisk不在{MIN_SHORT_RISK_PCT:.1f}-{MAX_SHORT_RISK_PCT:.1f}%"
        )

    # RR intentionally remains a warning/quality factor, not a hard gate.
    return failed


def classify_next_day(stock):
    failed = get_next_day_failed_gates(stock)

    hard_overheat = any(
        gate in failed
        for gate in ["RSI>=75", "MA20乖離>=12%", "突破延伸>8%"]
    )
    if hard_overheat:
        return "過熱／不追價"

    if not failed:
        return "明日進場候選"

    if (
        stock["next_day_score"] >= 68
        and stock["entry_quality_score"] >= 45
        and len(failed) <= 2
    ):
        return "等待明日確認"

    return "暫不考慮"

def classify_ready(stock):
    plan = stock["trade_plan"]

    if (
        stock["rsi"] >= 75
        or stock["distance_ma20_pct"] >= 12
        or stock["breakout_extension_pct"] > 8
        or stock["price"] > plan["chase_limit"]
    ):
        return "不追價"

    if plan["short_risk_pct"] > MAX_SHORT_RISK_PCT:
        return "風險偏高"

    if plan["real_risk_reward"] < MIN_REAL_RR:
        return "報酬空間不足"

    if stock["ready_score"] >= 80:
        if stock["price"] < plan["entry_low"]:
            return "等待轉強"
        if stock["volume_ratio"] < 1.2:
            return "等待量能"
        return "準備進場"

    if stock["ready_score"] >= 65:
        if stock["volume_ratio"] < 0.8:
            return "等待量能"
        return "持續觀察"

    return "尚未成熟"


def classify_mid_long(stock):
    if stock["mid_long_score"] >= 80:
        return "中長期趨勢強"
    if stock["mid_long_score"] >= 65:
        return "中長期持續追蹤"
    return "中長期一般"


def build_reasons(stock):
    reasons, risks = [], []

    if stock["ma5"] > stock["ma10"] > stock["ma20"] > 0:
        reasons.append("短期均線呈多頭排列")

    if stock["ma20"] > stock["ma60"] > 0:
        reasons.append("中期趨勢維持向上")

    d = stock["breakout_distance_pct"]

    if -1.5 <= d <= 2:
        reasons.append("位於20日突破附近的較佳觀察區")
    elif 2 < d <= 5:
        reasons.append("正在接近20日壓力區")

    if stock["volume_ratio"] >= 1.5:
        reasons.append(
            f"成交量放大至20日均量 {stock['volume_ratio']:.2f} 倍"
        )
    elif stock["volume_ratio"] >= 1.2:
        reasons.append("成交量開始擴張")

    if stock["rs20"] > 0:
        reasons.append(f"近20日相對大盤強 {stock['rs20']:.1f}%")

    if stock["best_sector"]:
        reasons.append(f"{stock['best_sector']}族群相對強勢")

    if stock["rsi"] >= 75:
        risks.append("RSI進入過熱區，不適合追價")
    elif stock["rsi"] >= 70:
        risks.append("RSI偏高，注意隔日追價風險")

    if stock["distance_ma20_pct"] >= 12:
        risks.append("股價與MA20乖離過大")

    if stock["breakout_extension_pct"] > 5:
        risks.append(
            f"突破後已延伸 {stock['breakout_extension_pct']:.1f}%"
        )

    if stock["volume_ratio"] < 0.8:
        risks.append("目前量能仍不足")

    if stock["breakout_distance_pct"] > 7:
        risks.append("距離突破位置仍較遠")

    plan = stock["trade_plan"]

    if plan["short_risk_pct"] > MAX_SHORT_RISK_PCT:
        risks.append(
            f"短線停損距離 {plan['short_risk_pct']:.1f}% 偏大"
        )

    if plan["real_risk_reward"] < MIN_REAL_RR:
        risks.append(
            f"目前風險報酬比 {plan['real_risk_reward']:.2f} 偏低"
        )

    if not reasons:
        reasons.append("目前以技術結構觀察為主")

    if not risks:
        risks.append("仍需觀察隔日開盤與成交量確認")

    return reasons[:5], risks[:4]


def build_trade_strategy(stock):
    p = stock["trade_plan"]
    price = stock["price"]

    lo = p["entry_low"]
    hi = p["entry_high"]
    breakout = p["breakout_price"]
    chase = p["chase_limit"]

    if (
        stock["rsi"] >= 75
        or stock["distance_ma20_pct"] >= 12
        or stock["breakout_extension_pct"] > 8
        or price > chase
    ):
        kind = "不追價"
        action = (
            f"現價 {price:.2f} 已偏離理想風險報酬區，"
            "等待回檔或重新形成買點。"
        )
    elif p["short_risk_pct"] > MAX_SHORT_RISK_PCT:
        kind = "風險過高"
        action = (
            f"目前技術停損距離約 {p['short_risk_pct']:.1f}%，"
            "超過短線風險上限，先不進場。"
        )
    elif p["real_risk_reward"] < MIN_REAL_RR:
        kind = "報酬空間不足"
        action = (
            f"目前風險報酬比約 {p['real_risk_reward']:.2f}，"
            "等待更好的價格或新的突破結構。"
        )
    elif lo <= price <= hi:
        kind = "回檔買進"
        action = (
            f"現價 {price:.2f} 位於理想買入區 "
            f"{lo:.2f}–{hi:.2f}；量價結構未轉弱時可分批評估。"
        )
    elif price < lo:
        kind = "等待轉強"
        action = (
            f"現價 {price:.2f} 低於理想買入區 "
            f"{lo:.2f}–{hi:.2f}；先等待止跌轉強。"
        )
    elif price < breakout:
        kind = "等待回檔"
        action = (
            f"現價 {price:.2f} 高於理想買入區 "
            f"{lo:.2f}–{hi:.2f}；不追價，等待回檔，"
            f"或突破 {breakout:.2f} 且放量後再評估。"
        )
    else:
        kind = "突破買進"
        action = (
            f"已站上突破價 {breakout:.2f}；若成交量同步放大且"
            f"價格不高於 {chase:.2f}，可視為突破型進場觀察。"
        )

    return {"type": kind, "action": action}


def build_next_day_action(stock):
    return stock["trade_strategy"]["action"]


# ============================================================
# SCAN
# ============================================================

def scan_one(code, market_ref):
    ticker, detected_market, df, download_error = download_history(code)

    if df is None:
        return None, {
            "code": code,
            "download_status": "failed",
            "filter_reason": download_error or "download_failed",
        }

    df = add_indicators(df)
    row = df.iloc[-1]
    price = safe_float(row["Close"])

    if price <= 0:
        return None, {
            "code": code,
            "download_status": "ok",
            "filter_reason": "invalid_price",
        }

    avg_value = safe_float(
        (df["Close"] * df["Volume"]).tail(20).mean()
    )

    eligibility_reasons = []
    if price < MIN_PRICE:
        eligibility_reasons.append("股價低於基本篩選門檻")
    if avg_value < MIN_AVG_DAILY_VALUE:
        eligibility_reasons.append("近20日平均成交金額低於基本篩選門檻")
    eligible_for_trade = not eligibility_reasons

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
    high60 = safe_float(row["HIGH60"])
    high120 = safe_float(row["HIGH120"])

    low10 = safe_float(row["LOW10"])
    low20 = safe_float(row["LOW20"])
    low60 = safe_float(row["LOW60"])

    ret20 = safe_float(row["RET20"])
    ret60 = safe_float(row["RET60"])

    trend_score, trend_reasons = calculate_trend_score(row)

    (
        breakout_score,
        breakout_reasons,
        breakout_distance,
        breakout_extension,
    ) = calculate_breakout_score(row)

    volume_score, volume_ratio, volume_reasons = calculate_volume_score(row)

    rs_score, rs20, rs60, rs_reasons = calculate_rs_score(row, market_ref)

    distance_ma20 = pct(price, ma20) if ma20 else 0

    stock = {
        "code": code,
        "name": info["name"],
        "market": market,
        "industry": info["industry"],
        "ticker": ticker,
        "groups": groups,
        "is_etf": False,
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
        "high60": r2(high60),
        "high120": r2(high120),
        "low10": r2(low10),
        "low20": r2(low20),
        "low60": r2(low60),
        "ret20": r2(ret20),
        "ret60": r2(ret60),
        "volume_ratio": r2(volume_ratio),
        "volume": int(safe_float(row["Volume"])),
        "volume_status": (
            "爆量" if volume_ratio >= 2
            else "放量" if volume_ratio >= 1.2
            else "正常" if volume_ratio >= 0.8
            else "量縮"
        ),
        "distance_ma20_pct": r2(distance_ma20),
        "breakout_distance_pct": r2(breakout_distance),
        "breakout_extension_pct": r2(breakout_extension),
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
        "avg_daily_value_20": r2(avg_value),
        "eligible_for_trade": eligible_for_trade,
        "eligibility_reasons": eligibility_reasons,
        "download_status": "ok",
        "filter_reason": "",
        "chart": build_chart_data(df),
        "news": [],
    }

    return stock, {
        "code": code,
        "download_status": "ok",
        "filter_reason": "",
    }


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(market):
    if not market:
        return {"status": "Neutral", "label": "Neutral"}

    price = safe_float(market.get("price"))
    ma20 = safe_float(market.get("ma20"))
    ma60 = safe_float(market.get("ma60"))
    ret20 = safe_float(market.get("ret20"))

    if price > ma20 > ma60 and ret20 > 0:
        return {"status": "Risk-On", "label": "Risk-On"}

    if price < ma20 and ret20 < -3:
        return {"status": "Risk-Off", "label": "Risk-Off"}

    return {"status": "Neutral", "label": "Neutral"}


# ============================================================
# DATA SAFETY
# ============================================================

def validate_scan_result(
    universe_count,
    download_ok_count,
    download_failed_count,
    valid_count,
    market_ref,
):
    print()
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    print(f"Universe        : {universe_count}")
    print(f"Download OK     : {download_ok_count}")
    print(f"Download Failed : {download_failed_count}")
    print(f"Eligible        : {valid_count}")

    if universe_count <= 0:
        raise RuntimeError(
            "CRITICAL: Stock universe is empty. Abort publication."
        )

    if download_ok_count <= 0:
        raise RuntimeError(
            "CRITICAL: No stock market data downloaded successfully."
        )

    download_success_rate = download_ok_count / universe_count

    print(f"Download Success: {download_success_rate:.1%}")

    if download_success_rate < MIN_DOWNLOAD_SUCCESS_RATE:
        raise RuntimeError(
            f"CRITICAL: Yahoo download success only "
            f"{download_ok_count}/{universe_count} "
            f"({download_success_rate:.1%}). "
            "Abort publication to protect previous data."
        )

    if valid_count == 0:
        raise RuntimeError(
            "CRITICAL: 0 eligible stocks after price/liquidity filters."
        )

    if not market_ref:
        raise RuntimeError(
            "CRITICAL: ^TWII market reference unavailable. "
            "Abort publication."
        )

    print("[VALIDATION OK] Market data is healthy.")
    return download_success_rate


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Taiwan Stock Radar V6.5 - TWSE Common Stocks Only")
    print("=" * 60)

    codes = get_twse_codes()
    print(f"Universe: {len(codes)}")

    market_ref = get_market_reference()
    print("Market:", market_ref)

    if not market_ref:
        raise RuntimeError(
            "CRITICAL: Unable to download ^TWII market data. "
            "Abort before scanning stocks."
        )

    raw_stocks = []
    scan_audit = []

    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}")

        try:
            stock, audit = scan_one(code, market_ref)
            scan_audit.append(audit)

            if stock:
                raw_stocks.append(stock)

        except Exception as e:
            print(f"[ERROR] {code}: {e}")
            scan_audit.append(
                {
                    "code": code,
                    "download_status": "failed",
                    "filter_reason": f"scan_exception:{type(e).__name__}",
                }
            )

    download_ok_count = sum(
        1 for x in scan_audit
        if x["download_status"] == "ok"
    )

    download_failed_count = len(codes) - download_ok_count

    filter_counts = {}
    for x in scan_audit:
        reason = x.get("filter_reason", "")
        if reason:
            filter_counts[reason] = filter_counts.get(reason, 0) + 1

    print(f"Downloaded stocks: {len(raw_stocks)}")
    print(f"Basic liquidity-eligible: {sum(1 for x in raw_stocks if x.get('eligible_for_trade'))}")

    download_success_rate = validate_scan_result(
        universe_count=len(codes),
        download_ok_count=download_ok_count,
        download_failed_count=download_failed_count,
        valid_count=len(raw_stocks),
        market_ref=market_ref,
    )

    institutional_map, institutional_meta = get_t86_history()
    print("[T86]", institutional_meta)

    # Sector membership map.
    group_members = {}
    trade_universe = [x for x in raw_stocks if x.get("eligible_for_trade", False)]
    for stock in trade_universe:
        for group in stock["groups"]:
            group_members.setdefault(group, []).append(stock)

    sector_map = build_sector_strength(trade_universe)

    for stock in raw_stocks:
        sector_score, best_sector, sector_detail = apply_sector_score(
            stock,
            sector_map,
            group_members,
        )

        stock["sector_score"] = sector_score
        stock["best_sector"] = best_sector
        stock["sector_detail"] = sector_detail

        institutional = institutional_map.get(stock["code"], {})
        stock["institutional_date"] = institutional_meta.get("date", "")
        stock["institutional_available"] = bool(institutional)

        institutional_fields = [
            "foreign_net_shares",
            "foreign_net_lots",
            "trust_net_shares",
            "trust_net_lots",
            "dealer_net_shares",
            "dealer_net_lots",
            "institutional_net_shares",
            "institutional_net_lots",
            "foreign_buy_streak",
            "foreign_sell_streak",
            "trust_buy_streak",
            "trust_sell_streak",
            "institutional_buy_streak",
            "institutional_sell_streak",
        ]
        for key in institutional_fields:
            stock[key] = institutional.get(key, 0)

        volume = stock.get("volume", 0)
        stock["institutional_volume_pct"] = (
            r2(stock["institutional_net_shares"] / volume * 100)
            if volume
            else 0
        )

        # Trade plan must exist before entry-quality and classification.
        stock["trade_plan"] = calculate_trade_plan(stock)

        stock["next_day_score"] = calculate_next_day_score(stock)
        stock["entry_quality_score"] = calculate_entry_quality_score(stock)
        stock["ready_score"] = calculate_ready_score(stock)
        stock["mid_long_score"] = calculate_mid_long_score(stock)

        stock["failed_gates"] = get_next_day_failed_gates(stock)
        stock["next_day_signal"] = classify_next_day(stock)
        stock["ready_signal"] = classify_ready(stock)
        stock["mid_long_signal"] = classify_mid_long(stock)

        stock["trade_strategy"] = build_trade_strategy(stock)

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
        trade_universe,
        key=lambda x: (
            x["next_day_score"],
            x["entry_quality_score"],
            x["volume_ratio"],
            x["rs20"],
        ),
        reverse=True,
    )

    ready = sorted(
        trade_universe,
        key=lambda x: (
            x["ready_score"],
            x["rs20"],
        ),
        reverse=True,
    )

    mid_long = sorted(
        trade_universe,
        key=lambda x: (
            x["mid_long_score"],
            x["rs60"],
        ),
        reverse=True,
    )

    radar = sorted(
        trade_universe,
        key=lambda x: (
            max(
                x["next_day_score"],
                x["ready_score"],
                x["mid_long_score"],
            ),
            x["next_day_score"],
            x["entry_quality_score"],
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
        if x["next_day_signal"] == "明日進場候選"
    ]

    # Do not pad strict next-day entries just to reach 10 names.
    used = {x["code"] for x in next_day_top}

    next_day_watch = [
        x for x in next_day
        if x["code"] not in used
        and x["next_day_signal"] == "等待明日確認"
    ][:20]

    ready_top = [
        x for x in ready
        if x["ready_signal"] in ["準備進場", "持續觀察", "等待量能", "等待轉強"]
    ][:READY_TOP]

    mid_long_top = [
        x for x in mid_long
        if x["mid_long_signal"] in ["中長期趨勢強", "中長期持續追蹤"]
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

    data_quality = {
        "universe_count": len(codes),
        "download_ok_count": download_ok_count,
        "download_failed_count": download_failed_count,
        "download_success_rate": r2(download_success_rate * 100),
        "eligible_count": sum(1 for x in raw_stocks if x.get("eligible_for_trade")),
        "filter_counts": filter_counts,
    }

    # Keep every TWSE/TPEx common stock searchable even when history is unavailable.
    scanned_codes = {x["code"] for x in raw_stocks}
    unavailable_stocks = []
    for code in codes:
        if code in scanned_codes:
            continue
        info = get_stock_info(code)
        exchange = "上櫃" if code in TPEX_CODES else "上市"
        unavailable_stocks.append({
            "code": code,
            "name": info.get("name") or code,
            "market": info.get("market") or exchange,
            "industry": info.get("industry") or "",
            "ticker": f"{code}.TWO" if code in TPEX_CODES else f"{code}.TW",
            "groups": get_groups(code) or ([info["industry"]] if info.get("industry") else []),
            "is_etf": False,
            "date": "",
            "price": 0,
            "eligible_for_trade": False,
            "eligibility_reasons": ["本次未取得足夠行情資料，暫無法分析"],
            "download_status": "failed",
            "download_error": "insufficient_or_unavailable_history",
            "next_day_score": 0,
            "ready_score": 0,
            "mid_long_score": 0,
            "next_day_signal": "資料不足",
            "ready_signal": "資料不足",
            "mid_long_signal": "資料不足",
            "recommendation_reasons": [],
            "risk_reasons": ["行情資料不足，沒有可用的K線與技術判讀"],
            "next_day_action": "本次未取得足夠行情資料，請稍後再查。",
            "trade_plan": {},
        })
    all_stock_rows = raw_stocks + unavailable_stocks

    # Keep the dashboard payload small enough for mobile browsers. 120-day
    # candles are written separately and fetched only when a chart is opened.
    def public_stocks(items):
        return [
            {key: value for key, value in stock.items() if key != "chart"}
            for stock in items
        ]

    payload = {
        "version": VERSION,
        "updated": now,
        "timezone": TIMEZONE,
        "strategy": (
            "V6.6 TWSE+TPEx Common-Stock Entry & Position Radar: "
            "Trend + Breakout + Volume + Relative Strength "
            "+ Sector Strength + Entry Quality + Risk/Reward "
            "+ Overheat Control"
        ),
        "universe_count": len(codes),
        "valid_count": sum(1 for x in raw_stocks if x.get("eligible_for_trade")),
        "data_quality": data_quality,
        "market_regime": market_regime,
        "institutional_data": institutional_meta,
        "market_reference": {
            k: r2(v)
            for k, v in market_ref.items()
        },
        "signal_counts": signal_counts,
        "sector_ranking": sector_ranking,
        "next_day_top": public_stocks(next_day_top),
        "next_day_watch": public_stocks(next_day_watch),
        "ready_top": public_stocks(ready_top),
        "mid_long_top": public_stocks(mid_long_top),
        "stocks": public_stocks(radar_top),
        "all_stocks": public_stocks(all_stock_rows),
    }

    os.makedirs("docs", exist_ok=True)
    os.makedirs("docs/charts", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for stock in raw_stocks:
        code = str(stock["code"])
        if not code.isdigit():
            continue
        with open(
            os.path.join("docs", "charts", f"{code}.json"),
            "w",
            encoding="utf-8",
        ) as chart_file:
            json.dump(
                stock.get("chart", []),
                chart_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    with open(
        "docs/data.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
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
        row["failed_gates"] = " | ".join(
            stock.get("failed_gates", [])
        )

        for key, value in stock["trade_plan"].items():
            row[key] = value

        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        "data/signals.csv",
        index=False,
        encoding="utf-8",
    )

    print("[SAVE OK] data/signals.csv")

    print()
    print("=" * 60)
    print(f"{VERSION} COMPLETE")
    print("=" * 60)

    print("Updated:", now)
    print("Market:", market_regime["label"])
    print("Universe:", len(codes))
    print("Download OK:", download_ok_count)
    print("Download Failed:", download_failed_count)
    print("Scanned stocks:", len(raw_stocks))
    print("Trade-eligible stocks:", sum(1 for x in raw_stocks if x.get("eligible_for_trade")))
    print(f"Download Success: {download_success_rate:.1%}")

    print()
    print("Tomorrow Top Picks:")

    if not next_day_top:
        print(f"No stock passed all strict {VERSION} tomorrow-entry gates.")

    for stock in next_day_top:
        plan = stock["trade_plan"]

        print(
            f"{stock['code']} "
            f"{stock['name']} "
            f"Score={stock['next_day_score']} "
            f"EntryQ={stock['entry_quality_score']} "
            f"Entry={plan['entry_low']}-{plan['entry_high']} "
            f"Breakout={plan['breakout_price']} "
            f"Stop={plan['short_stop']} "
            f"Risk={plan['short_risk_pct']}% "
            f"RR={plan['real_risk_reward']}"
        )

    print()
    print("Tomorrow Watch:")

    if not next_day_watch:
        print("No near-pass stock with only 1-2 failed gates.")

    for stock in next_day_watch:
        plan = stock["trade_plan"]
        failed_text = ", ".join(stock.get("failed_gates", [])) or "None"
        print(
            f"{stock['code']} "
            f"{stock['name']} "
            f"Score={stock['next_day_score']} "
            f"EntryQ={stock['entry_quality_score']} "
            f"Entry={plan['entry_low']}-{plan['entry_high']} "
            f"Stop={plan['short_stop']} "
            f"Risk={plan['short_risk_pct']}% "
            f"RR={plan['real_risk_reward']} "
            f"Failed={failed_text}"
        )

    print()
    print("Top Sectors:")

    for sector in sector_ranking[:10]:
        print(
            f"#{sector['rank']} "
            f"{sector['group']} "
            f"Score={sector['score']} "
            f"Members={sector['count']}"
        )


if __name__ == "__main__":
    main()

