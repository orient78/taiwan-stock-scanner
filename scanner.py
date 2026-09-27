import os
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from stock_pool import get_stock_codes, get_groups


# =========================================================
# V5 CONFIG
# =========================================================

VERSION = "V5"
TIMEZONE = "Asia/Taipei"

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000

# V5: keep more candidates for frontend filtering
MAX_RESULTS = 80

# Keep candidates with at least some technical structure
MIN_BASE_SCORE = 2

# Daily K-line data sent to frontend
CHART_DAYS = 90


# =========================================================
# Utility
# =========================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def round2(value):
    return round(safe_float(value), 2)


# =========================================================
# Indicator calculation
# =========================================================

def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calculate_macd(close):
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    hist = macd - signal

    return macd, signal, hist


def calculate_atr(df, period=14):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


def add_indicators(df):

    df = df.copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    (
        df["MACD"],
        df["MACD_SIGNAL"],
        df["MACD_HIST"],
    ) = calculate_macd(df["Close"])

    df["VOL20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Previous 20 trading-day high.
    # Shift(1) avoids comparing today's price with today's own high.
    df["HIGH20"] = (
        df["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["ATR14"] = calculate_atr(
        df,
        14
    )

    return df


# =========================================================
# Launch Score V5
# =========================================================

def calculate_launch_score(
    price,
    ma5,
    ma10,
    ma20,
    ma60,
    ma20_prev,
    rsi,
    macd,
    macd_signal,
    hist,
    hist_prev,
    volume_ratio,
    high20,
):

    score = 0
    reasons = []

    # -----------------------------------------------------
    # 1. Trend structure — 25 pts
    # -----------------------------------------------------

    if price > ma20:
        score += 7
        reasons.append("股價站上 MA20")

    if ma5 > ma10:
        score += 6
        reasons.append("MA5 > MA10")

    if ma10 > ma20:
        score += 6
        reasons.append("MA10 > MA20")

    if ma20 > ma60 and ma20 > ma20_prev:
        score += 6
        reasons.append("MA20 > MA60 且 MA20 上彎")

    # -----------------------------------------------------
    # 2. Momentum — 20 pts
    # -----------------------------------------------------

    if 50 <= rsi <= 65:
        score += 8
        reasons.append(
            f"RSI {rsi:.1f} 位於健康偏多區"
        )

    elif 65 < rsi <= 70:
        score += 5
        reasons.append(
            f"RSI {rsi:.1f} 動能偏強"
        )

    elif 45 <= rsi < 50:
        score += 3

    if macd > macd_signal:
        score += 6
        reasons.append("MACD 位於 Signal 上方")

    if hist > hist_prev:
        score += 6
        reasons.append("MACD 柱狀體增強")

    # -----------------------------------------------------
    # 3. Breakout readiness — 20 pts
    # -----------------------------------------------------

    if high20 > 0:

        breakout_distance = (
            (high20 - price) / high20
        ) * 100

        if price >= high20:
            score += 20
            reasons.append("已突破 20 日高點")

        elif breakout_distance <= 2:
            score += 17
            reasons.append(
                f"距 20 日突破價僅 {breakout_distance:.1f}%"
            )

        elif breakout_distance <= 5:
            score += 12
            reasons.append(
                f"距 20 日突破價 {breakout_distance:.1f}%"
            )

        elif breakout_distance <= 8:
            score += 6

    # -----------------------------------------------------
    # 4. Volume expansion — 20 pts
    # -----------------------------------------------------

    if volume_ratio >= 2:
        score += 20
        reasons.append(
            f"量比 {volume_ratio:.2f}x，明顯放量"
        )

    elif volume_ratio >= 1.5:
        score += 17
        reasons.append(
            f"量比 {volume_ratio:.2f}x，量能擴張"
        )

    elif volume_ratio >= 1.2:
        score += 12
        reasons.append(
            f"量比 {volume_ratio:.2f}x"
        )

    elif volume_ratio >= 1:
        score += 7

    # -----------------------------------------------------
    # 5. MA convergence / launch structure — 15 pts
    # -----------------------------------------------------

    mas = [
        x for x in [ma5, ma10, ma20]
        if x > 0
    ]

    if len(mas) == 3:

        ma_spread = (
            (max(mas) - min(mas))
            / price
            * 100
        )

        if ma_spread <= 1.5:
            score += 10
            reasons.append("短期均線高度收斂")

        elif ma_spread <= 3:
            score += 7

        elif ma_spread <= 5:
            score += 3

    if ma5 > ma10 > ma20:
        score += 5
        reasons.append(
            "MA5 > MA10 > MA20 多頭排列"
        )

    return min(score, 100), reasons


# =========================================================
# Confidence
# =========================================================

def calculate_confidence(
    price,
    ma5,
    ma10,
    ma20,
    ma60,
    rsi,
    macd,
    macd_signal,
    hist,
    hist_prev,
    volume_ratio,
    high20,
):

    confidence = 0

    if price > ma5:
        confidence += 10

    if price > ma10:
        confidence += 10

    if price > ma20:
        confidence += 10

    if ma5 > ma10 > ma20:
        confidence += 15

    if ma20 > ma60:
        confidence += 10

    if 50 <= rsi <= 70:
        confidence += 10

    if macd > macd_signal:
        confidence += 10

    if hist > hist_prev:
        confidence += 10

    if volume_ratio >= 1.2:
        confidence += 10

    if high20 > 0:

        if price >= high20:
            confidence += 5

        elif price >= high20 * 0.97:
            confidence += 3

    return min(confidence, 100)


# =========================================================
# V5 detailed recommendation reasons
# =========================================================

def build_recommendation_reasons(
    price,
    ma5,
    ma10,
    ma20,
    ma60,
    rsi,
    macd,
    macd_signal,
    hist,
    hist_prev,
    volume_ratio,
    high20,
):

    positive = []
    risks = []

    # -----------------------------------------------------
    # MA structure
    # -----------------------------------------------------

    if (
        price > ma5
        and price > ma10
        and price > ma20
    ):
        positive.append(
            "股價同時站上 MA5 / MA10 / MA20，短線結構偏多"
        )

    elif price > ma20:
        positive.append(
            "股價仍站在 MA20 之上"
        )

    else:
        risks.append(
            "股價尚未站穩 MA20"
        )

    if ma5 > ma10 > ma20:
        positive.append(
            "MA5 > MA10 > MA20，三條短期均線呈多頭排列"
        )

    else:
        risks.append(
            "短期均線尚未形成完整 MA5 > MA10 > MA20 多頭排列"
        )

    if ma20 > ma60:
        positive.append(
            "MA20 位於 MA60 之上，中期趨勢維持偏多"
        )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if 50 <= rsi <= 65:
        positive.append(
            f"RSI14 為 {rsi:.1f}，位於健康偏多區間"
        )

    elif 65 < rsi < 75:
        positive.append(
            f"RSI14 為 {rsi:.1f}，動能強但已接近高檔"
        )

    elif rsi >= 75:
        risks.append(
            f"RSI14 已達 {rsi:.1f}，短線有過熱風險"
        )

    elif rsi < 45:
        risks.append(
            f"RSI14 僅 {rsi:.1f}，目前動能偏弱"
        )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if macd > macd_signal:

        if hist > hist_prev:
            positive.append(
                "MACD 位於 Signal 上方，且柱狀體持續增強"
            )

        else:
            positive.append(
                "MACD 仍位於 Signal 上方"
            )

    else:
        risks.append(
            "MACD 尚未站上 Signal，動能仍需確認"
        )

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    if volume_ratio >= 1.5:
        positive.append(
            f"成交量為 20 日均量 {volume_ratio:.2f} 倍，出現明顯放量"
        )

    elif volume_ratio >= 1.2:
        positive.append(
            f"成交量為 20 日均量 {volume_ratio:.2f} 倍，量能開始增溫"
        )

    elif volume_ratio < 1:
        risks.append(
            f"量比僅 {volume_ratio:.2f}x，突破仍缺乏量能確認"
        )

    # -----------------------------------------------------
    # Breakout
    # -----------------------------------------------------

    breakout_distance_pct = 0

    if high20 > 0:

        breakout_distance_pct = (
            (high20 - price)
            / high20
            * 100
        )

        if price >= high20:
            positive.append(
                f"股價已突破 20 日高點 {high20:.2f}"
            )

        elif breakout_distance_pct <= 2:
            positive.append(
                f"距 20 日突破價 {high20:.2f} 僅 "
                f"{breakout_distance_pct:.1f}%"
            )

        elif breakout_distance_pct <= 5:
            positive.append(
                f"距 20 日高點約 {breakout_distance_pct:.1f}%，"
                "接近突破觀察區"
            )

        else:
            risks.append(
                f"距 20 日突破價仍有 "
                f"{breakout_distance_pct:.1f}%"
            )

    # -----------------------------------------------------
    # MA20 over-extension
    # -----------------------------------------------------

    distance_ma20_pct = 0

    if ma20 > 0:

        distance_ma20_pct = (
            (price - ma20)
            / ma20
            * 100
        )

        if distance_ma20_pct >= 12:
            risks.append(
                f"股價高於 MA20 約 {distance_ma20_pct:.1f}%，"
                "乖離過大，不宜追價"
            )

        elif distance_ma20_pct >= 8:
            risks.append(
                f"股價高於 MA20 約 {distance_ma20_pct:.1f}%，"
                "短線乖離偏高"
            )

    return (
        positive,
        risks,
        breakout_distance_pct,
        distance_ma20_pct,
    )


# =========================================================
# Signal classification
# =========================================================

def classify_signal(
    launch_score,
    confidence,
    price,
    high20,
    rsi,
    distance_ma20_pct,
):

    overheated = (
        rsi >= 75
        or distance_ma20_pct >= 12
    )

    if overheated:
        return "過熱／不追價", True

    near_breakout = (
        high20 > 0
        and price >= high20 * 0.98
    )

    if (
        launch_score >= 80
        and confidence >= 75
        and near_breakout
    ):
        return "符合進場條件", False

    if (
        launch_score >= 65
        and confidence >= 60
    ):
        return "等待突破確認", False

    return "條件不足", False


# =========================================================
# Star rating
# =========================================================

def confidence_stars(confidence):

    if confidence >= 90:
        return 5

    if confidence >= 80:
        return 4

    if confidence >= 70:
        return 3

    if confidence >= 60:
        return 2

    return 1


# =========================================================
# Base V3/V4 score
# =========================================================

def calculate_base_score(
    price,
    ma5,
    ma10,
    ma20,
    ma60,
    ma20_prev,
    rsi,
    macd,
    macd_signal,
    volume_ratio,
    high20,
):

    score = 0
    reasons = []

    above_3ma = (
        price > ma5
        and price > ma10
        and price > ma20
    )

    trend = (
        price > ma20
        and ma20 > ma60
        and ma20 > ma20_prev
    )

    rsi_ok = (
        50 <= rsi <= 70
    )

    macd_ok = (
        macd > macd_signal
    )

    breakout = (
        high20 > 0
        and price > high20
    )

    volume_ok = (
        volume_ratio >= 1.5
    )

    if above_3ma:
        score += 1
        reasons.append(
            "站上 MA5 / MA10 / MA20"
        )

    if trend:
        score += 1
        reasons.append(
            "MA20 > MA60 且趨勢向上"
        )

    if rsi_ok:
        score += 1
        reasons.append(
            f"RSI {rsi:.1f}"
        )

    if macd_ok:
        score += 1
        reasons.append(
            "MACD 多方"
        )

    if breakout:
        score += 1
        reasons.append(
            "突破 20 日高點"
        )

    if volume_ok:
        score += 1
        reasons.append(
            f"量比 {volume_ratio:.2f}x"
        )

    return score, reasons, above_3ma


# =========================================================
# Category
# =========================================================

def get_category(
    price,
    high20,
    ma20,
    ma60,
):

    if (
        high20 > 0
        and price >= high20
    ):
        return "強勢突破"

    if (
        price > ma20
        and ma20 > ma60
    ):
        return "多頭趨勢"

    return "觀察名單"


# =========================================================
# Chart data
# =========================================================

def build_chart_data(df):

    chart_df = (
        df.tail(CHART_DAYS)
        .copy()
    )

    result = []

    for index, row in chart_df.iterrows():

        result.append({
            "date": index.strftime("%Y-%m-%d"),
            "open": round2(row["Open"]),
            "high": round2(row["High"]),
            "low": round2(row["Low"]),
            "close": round2(row["Close"]),
            "volume": int(
                safe_float(row["Volume"])
            ),
            "ma5": round2(row["MA5"]),
            "ma10": round2(row["MA10"]),
            "ma20": round2(row["MA20"]),
            "ma60": round2(row["MA60"]),
        })

    return result


# =========================================================
# Stock name
# =========================================================

def get_stock_name(code, ticker):

    # Keep the scanner robust:
    # Yahoo metadata occasionally fails.
    try:
        info = yf.Ticker(ticker).fast_info

        # fast_info normally does not include name.
        # Keep fallback to code and let frontend display code
        # if metadata is unavailable.
    except Exception:
        pass

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.get_info()

        name = (
            info.get("shortName")
            or info.get("longName")
        )

        if name:
            return str(name)

    except Exception:
        pass

    return code


# =========================================================
# Scan one stock
# =========================================================

def scan_one(code):

    # Most symbols in the pool are TWSE.
    ticker = f"{code}.TW"

    try:

        df = yf.download(
            ticker,
            period="9mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        # Try OTC ticker when TWSE fails
        if df is None or len(df) < 65:

            ticker = f"{code}.TWO"

            df = yf.download(
                ticker,
                period="9mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

        if df is None or len(df) < 65:
            return None

        # yfinance sometimes returns MultiIndex
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required:
            if column not in df.columns:
                return None

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

        if len(df) < 65:
            return None

        df = add_indicators(df)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        price = safe_float(
            latest["Close"]
        )

        ma5 = safe_float(
            latest["MA5"]
        )

        ma10 = safe_float(
            latest["MA10"]
        )

        ma20 = safe_float(
            latest["MA20"]
        )

        ma60 = safe_float(
            latest["MA60"]
        )

        ma20_prev = safe_float(
            previous["MA20"]
        )

        rsi = safe_float(
            latest["RSI14"]
        )

        macd = safe_float(
            latest["MACD"]
        )

        macd_signal = safe_float(
            latest["MACD_SIGNAL"]
        )

        hist = safe_float(
            latest["MACD_HIST"]
        )

        hist_prev = safe_float(
            previous["MACD_HIST"]
        )

        volume = safe_float(
            latest["Volume"]
        )

        vol20 = safe_float(
            latest["VOL20"]
        )

        high20 = safe_float(
            latest["HIGH20"]
        )

        atr14 = safe_float(
            latest["ATR14"]
        )

        if (
            price <= 0
            or ma20 <= 0
            or ma60 <= 0
        ):
            return None

        groups = get_groups(code)

        is_etf = (
            "ETF" in groups
        )

        # -------------------------------------------------
        # Liquidity
        # -------------------------------------------------

        avg_daily_value = safe_float(
            (
                df["Close"]
                * df["Volume"]
            )
            .tail(20)
            .mean()
        )

        if not is_etf:

            if price < MIN_PRICE:
                return None

            if (
                avg_daily_value
                < MIN_AVG_DAILY_VALUE
            ):
                return None

        volume_ratio = (
            volume / vol20
            if vol20 > 0
            else 0
        )

        # -------------------------------------------------
        # Base score
        # -------------------------------------------------

        (
            base_score,
            base_reasons,
            above_3ma,
        ) = calculate_base_score(
            price,
            ma5,
            ma10,
            ma20,
            ma60,
            ma20_prev,
            rsi,
            macd,
            macd_signal,
            volume_ratio,
            high20,
        )

        if base_score < MIN_BASE_SCORE:
            return None

        # -------------------------------------------------
        # Launch Score
        # -------------------------------------------------

        (
            launch_score,
            launch_reasons,
        ) = calculate_launch_score(
            price,
            ma5,
            ma10,
            ma20,
            ma60,
            ma20_prev,
            rsi,
            macd,
            macd_signal,
            hist,
            hist_prev,
            volume_ratio,
            high20,
        )

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = calculate_confidence(
            price,
            ma5,
            ma10,
            ma20,
            ma60,
            rsi,
            macd,
            macd_signal,
            hist,
            hist_prev,
            volume_ratio,
            high20,
        )

        # -------------------------------------------------
        # Human-readable V5 reasons
        # -------------------------------------------------

        (
            recommendation_reasons,
            risk_reasons,
            breakout_distance_pct,
            distance_ma20_pct,
        ) = build_recommendation_reasons(
            price,
            ma5,
            ma10,
            ma20,
            ma60,
            rsi,
            macd,
            macd_signal,
            hist,
            hist_prev,
            volume_ratio,
            high20,
        )

        # -------------------------------------------------
        # Signal
        # -------------------------------------------------

        signal, overheated = classify_signal(
            launch_score,
            confidence,
            price,
            high20,
            rsi,
            distance_ma20_pct,
        )

        # -------------------------------------------------
        # Entry / risk reference
        # -------------------------------------------------

        entry_low = max(
            ma10,
            ma20
        )

        if high20 > 0:
            entry_high = min(
                price,
                high20
            )
        else:
            entry_high = price

        if entry_high < entry_low:
            entry_high = price

        risk_reference = (
            price - 2 * atr14
            if atr14 > 0
            else ma20
        )

        category = get_category(
            price,
            high20,
            ma20,
            ma60,
        )

        # -------------------------------------------------
        # Observation trigger
        # -------------------------------------------------

        if high20 > 0 and price < high20:

            trigger_text = (
                f"觀察是否帶量突破 "
                f"{high20:.2f}，"
                f"量比最好 > 1.5x"
            )

        elif price >= high20 > 0:

            trigger_text = (
                "已突破 20 日高點，"
                "觀察突破後能否守穩且量能延續"
            )

        else:

            trigger_text = (
                "等待短均線多頭排列與量能同步轉強"
            )

        # -------------------------------------------------
        # Date
        # -------------------------------------------------

        latest_date = (
            df.index[-1]
            .strftime("%Y-%m-%d")
        )

        # -------------------------------------------------
        # Stock name
        # -------------------------------------------------

        name = get_stock_name(
            code,
            ticker
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        return {

            "code": code,
            "name": name,
            "market": (
                "上櫃"
                if ticker.endswith(".TWO")
                else "上市"
            ),
            "ticker": ticker,

            "groups": groups,
            "is_etf": is_etf,

            "date": latest_date,
            "price": round2(price),

            # Base technical score
            "score": base_score,
            "category": category,
            "above_3ma": above_3ma,
            "reasons": base_reasons,

            # Indicators
            "rsi": round2(rsi),
            "volume_ratio": round2(
                volume_ratio
            ),

            "ma5": round2(ma5),
            "ma10": round2(ma10),
            "ma20": round2(ma20),
            "ma60": round2(ma60),

            "macd": round2(macd),
            "macd_signal": round2(
                macd_signal
            ),
            "macd_hist": round2(hist),

            "atr14": round2(atr14),

            # V5
            "launch_score": int(
                launch_score
            ),

            "confidence": int(
                confidence
            ),

            "stars": confidence_stars(
                confidence
            ),

            "signal": signal,

            "overheated": overheated,

            "distance_ma20_pct": round2(
                distance_ma20_pct
            ),

            "breakout_distance_pct": round2(
                breakout_distance_pct
            ),

            "breakout_price": round2(
                high20
            ),

            "entry_low": round2(
                entry_low
            ),

            "entry_high": round2(
                entry_high
            ),

            "risk_reference": round2(
                risk_reference
            ),

            # Detailed explanation
            "launch_reasons": (
                launch_reasons
            ),

            "recommendation_reasons": (
                recommendation_reasons
            ),

            "risk_reasons": (
                risk_reasons
            ),

            "trigger": trigger_text,

            # News placeholder.
            # V5 frontend will show this section only
            # when news is available.
            "news": [],

            # Daily candlestick data
            "chart": build_chart_data(df),
        }

    except Exception as e:

        print(
            f"[ERROR] {code}: {e}"
        )

        return None


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Taiwan Stock Scanner V5")
    print("=" * 60)

    codes = get_stock_codes()

    print(
        f"Universe: {len(codes)} symbols"
    )

    results = []

    for index, code in enumerate(
        codes,
        start=1
    ):

        print(
            f"[{index}/{len(codes)}] "
            f"Scanning {code}"
        )

        stock = scan_one(code)

        if stock:
            results.append(stock)

    # =====================================================
    # V5 Ranking
    # =====================================================

    results.sort(
        key=lambda x: (
            x["launch_score"],
            x["confidence"],
            x["score"],
            x["volume_ratio"],
        ),
        reverse=True,
    )

    # Keep Top 80
    results = results[:MAX_RESULTS]

    # Add ranking
    for rank, stock in enumerate(
        results,
        start=1
    ):
        stock["rank"] = rank

    # =====================================================
    # Group counts
    # =====================================================

    group_counts = {}

    for stock in results:

        for group in stock["groups"]:

            group_counts[group] = (
                group_counts.get(
                    group,
                    0
                )
                + 1
            )

    group_counts = dict(
        sorted(
            group_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # =====================================================
    # Signal counts
    # =====================================================

    signal_names = [
        "符合進場條件",
        "等待突破確認",
        "過熱／不追價",
        "條件不足",
    ]

    signal_counts = {
        name: 0
        for name in signal_names
    }

    for stock in results:

        signal = stock["signal"]

        signal_counts[signal] = (
            signal_counts.get(
                signal,
                0
            )
            + 1
        )

    # =====================================================
    # Payload
    # =====================================================

    payload = {

        "version": VERSION,

        "updated": datetime.now(
            ZoneInfo(TIMEZONE)
        ).strftime(
            "%Y-%m-%d %H:%M"
        ),

        "timezone": TIMEZONE,

        "universe_count": len(codes),

        "strategy": (
            "V5 Launch Radar: "
            "MA structure + RSI + MACD momentum + "
            "20D breakout readiness + volume expansion + "
            "overheat control + detailed recommendation reasons"
        ),

        "count": len(results),

        "group_counts": group_counts,

        "signal_counts": signal_counts,

        "stocks": results,
    }

    # =====================================================
    # Output folders
    # =====================================================

    os.makedirs(
        "docs",
        exist_ok=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    # =====================================================
    # JSON
    # =====================================================

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

    # =====================================================
    # CSV
    # =====================================================

    csv_rows = []

    for stock in results:

        row = stock.copy()

        # Don't put huge chart arrays into CSV
        row.pop(
            "chart",
            None
        )

        row["groups"] = " / ".join(
            stock["groups"]
        )

        row["reasons"] = " | ".join(
            stock["reasons"]
        )

        row["launch_reasons"] = " | ".join(
            stock["launch_reasons"]
        )

        row[
            "recommendation_reasons"
        ] = " | ".join(
            stock["recommendation_reasons"]
        )

        row["risk_reasons"] = " | ".join(
            stock["risk_reasons"]
        )

        row["news"] = ""

        csv_rows.append(row)

    if csv_rows:

        pd.DataFrame(
            csv_rows
        ).to_csv(
            "data/signals.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 60)
    print("V5 Scan Complete")
    print("=" * 60)

    print(
        "Universe:",
        len(codes)
    )

    print(
        "Selected:",
        len(results)
    )

    print()

    print("Signal Summary:")

    for signal, count in (
        signal_counts.items()
    ):

        print(
            f"  {signal}: {count}"
        )

    print()

    print("Top 10:")

    for stock in results[:10]:

        print(
            f'#{stock["rank"]:02d} '
            f'{stock["code"]} '
            f'{stock["name"]} | '
            f'Launch {stock["launch_score"]} | '
            f'Confidence {stock["confidence"]} | '
            f'{stock["signal"]}'
        )


if __name__ == "__main__":
    main()
