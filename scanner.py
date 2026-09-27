import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import twstock
import yfinance as yf

from stock_pool import get_stock_codes, get_groups


TZ = ZoneInfo("Asia/Taipei")

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000
MAX_RESULTS = 50
MIN_SCORE = 3

CHART_DAYS = 60


def get_universe():

    stocks = []

    for code in get_stock_codes():

        info = twstock.codes.get(code)

        if info is None:
            print(f"Unknown code: {code}")
            continue

        groups = get_groups(code)

        is_etf = "ETF" in groups

        if info.market == "上市":
            suffix = ".TW"

        elif info.market == "上櫃":
            suffix = ".TWO"

        else:
            print(f"Unsupported market: {code}")
            continue

        stocks.append(
            (
                code + suffix,
                code,
                info.name,
                info.market,
                groups,
                is_etf,
            )
        )

    return stocks


# ==========================================================
# Indicators
# ==========================================================

def add_indicators(df):

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Moving averages
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    # RSI 14
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI14"] = (
        100 -
        (100 / (1 + rs))
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_HIST"] = (
        df["MACD"]
        - df["MACD_SIGNAL"]
    )

    # Volume
    df["VOL20"] = (
        volume
        .rolling(20)
        .mean()
    )

    # Previous 20-day high
    df["HIGH20_PREV"] = (
        close
        .shift(1)
        .rolling(20)
        .max()
    )

    # ATR 14
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR14"] = (
        tr
        .rolling(14)
        .mean()
    )

    return df


# ==========================================================
# V4 Launch Score
# ==========================================================

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
    macd_hist,
    macd_hist_prev,
    high20,
    volume_ratio,
):

    score = 0
    reasons = []

    # ------------------------------------------------------
    # 1. Trend structure - 25
    # ------------------------------------------------------

    if price > ma20:
        score += 7
        reasons.append("股價站上MA20")

    if ma5 > ma10:
        score += 6
        reasons.append("MA5站上MA10")

    if ma10 > ma20:
        score += 6
        reasons.append("MA10站上MA20")

    if ma20 > ma60 and ma20 > ma20_prev:
        score += 6
        reasons.append("中期趨勢向上")

    # ------------------------------------------------------
    # 2. Momentum - 20
    # ------------------------------------------------------

    if 50 <= rsi <= 65:
        score += 8
        reasons.append("RSI健康偏多")

    elif 65 < rsi <= 70:
        score += 5
        reasons.append("RSI偏強")

    elif 45 <= rsi < 50:
        score += 3

    if macd > macd_signal:
        score += 6
        reasons.append("MACD多方")

    if macd_hist > macd_hist_prev:
        score += 6
        reasons.append("MACD動能增強")

    # ------------------------------------------------------
    # 3. Breakout readiness - 20
    # ------------------------------------------------------

    if high20 > 0:

        breakout_distance = (
            (high20 - price)
            / high20
            * 100
        )

        if price > high20:
            score += 20
            reasons.append("突破20日高點")

        elif 0 <= breakout_distance <= 2:
            score += 17
            reasons.append("距突破不到2%")

        elif 2 < breakout_distance <= 5:
            score += 12
            reasons.append("接近20日突破")

        elif 5 < breakout_distance <= 8:
            score += 6

    # ------------------------------------------------------
    # 4. Volume / price - 20
    # ------------------------------------------------------

    if volume_ratio >= 2:
        score += 20
        reasons.append("量能明顯放大")

    elif volume_ratio >= 1.5:
        score += 17
        reasons.append("成交量突破")

    elif volume_ratio >= 1.2:
        score += 12
        reasons.append("量能升溫")

    elif volume_ratio >= 1.0:
        score += 7

    # ------------------------------------------------------
    # 5. MA convergence / early launch - 15
    # ------------------------------------------------------

    ma_values = [
        ma5,
        ma10,
        ma20,
    ]

    ma_spread = (
        (
            max(ma_values)
            - min(ma_values)
        )
        / ma20
        * 100
    )

    if ma_spread <= 1.5:
        score += 10
        reasons.append("短均線高度收斂")

    elif ma_spread <= 3:
        score += 7
        reasons.append("短均線收斂")

    elif ma_spread <= 5:
        score += 3

    if ma5 > ma10 > ma20:
        score += 5
        reasons.append("短均線轉多")

    return min(
        int(round(score)),
        100
    ), reasons


# ==========================================================
# V4 Confidence
# ==========================================================

def calculate_confidence(
    price,
    ma5,
    ma10,
    ma20,
    ma60,
    rsi,
    macd,
    macd_signal,
    macd_hist,
    macd_hist_prev,
    volume_ratio,
    high20,
):

    points = 0

    # Trend consistency
    if price > ma5:
        points += 10

    if price > ma10:
        points += 10

    if price > ma20:
        points += 10

    if ma5 > ma10 > ma20:
        points += 15

    if ma20 > ma60:
        points += 10

    # Momentum consistency
    if 50 <= rsi <= 70:
        points += 10

    if macd > macd_signal:
        points += 10

    if macd_hist > macd_hist_prev:
        points += 10

    # Volume confirmation
    if volume_ratio >= 1.2:
        points += 10

    # Near / above breakout
    if high20 > 0:

        if price >= high20:
            points += 5

        elif price >= high20 * 0.97:
            points += 3

    return min(
        int(points),
        100
    )


def confidence_stars(confidence):

    if confidence >= 90:
        return 5

    if confidence >= 75:
        return 4

    if confidence >= 60:
        return 3

    if confidence >= 40:
        return 2

    return 1


# ==========================================================
# V4 Signal
# ==========================================================

def determine_signal(
    launch_score,
    confidence,
    rsi,
    distance_ma20_pct,
    price,
    high20,
):

    overheated = bool(
        rsi >= 75
        or distance_ma20_pct >= 12
    )

    if overheated:

        return (
            "過熱／不追價",
            True
        )

    if (
        launch_score >= 80
        and confidence >= 75
        and price >= high20 * 0.98
    ):

        return (
            "符合進場條件",
            False
        )

    if (
        launch_score >= 65
        and confidence >= 60
    ):

        return (
            "等待突破確認",
            False
        )

    return (
        "條件不足",
        False
    )


# ==========================================================
# Chart data
# ==========================================================

def build_chart_data(df):

    chart_df = df.tail(
        CHART_DAYS
    )

    result = []

    for index, row in chart_df.iterrows():

        result.append(
            {
                "date":
                    pd.Timestamp(
                        index
                    ).strftime(
                        "%Y-%m-%d"
                    ),

                "open":
                    round(
                        float(
                            row["Open"]
                        ),
                        2
                    ),

                "high":
                    round(
                        float(
                            row["High"]
                        ),
                        2
                    ),

                "low":
                    round(
                        float(
                            row["Low"]
                        ),
                        2
                    ),

                "close":
                    round(
                        float(
                            row["Close"]
                        ),
                        2
                    ),

                "volume":
                    int(
                        float(
                            row["Volume"]
                        )
                    ),

                "ma5":
                    (
                        round(
                            float(
                                row["MA5"]
                            ),
                            2
                        )
                        if pd.notna(
                            row["MA5"]
                        )
                        else None
                    ),

                "ma10":
                    (
                        round(
                            float(
                                row["MA10"]
                            ),
                            2
                        )
                        if pd.notna(
                            row["MA10"]
                        )
                        else None
                    ),

                "ma20":
                    (
                        round(
                            float(
                                row["MA20"]
                            ),
                            2
                        )
                        if pd.notna(
                            row["MA20"]
                        )
                        else None
                    ),

                "ma60":
                    (
                        round(
                            float(
                                row["MA60"]
                            ),
                            2
                        )
                        if pd.notna(
                            row["MA60"]
                        )
                        else None
                    ),
            }
        )

    return result


# ==========================================================
# Scan
# ==========================================================

def scan_one(
    ticker,
    code,
    name,
    market,
    groups,
    is_etf,
):

    try:

        df = yf.download(
            ticker,
            period="9mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=15,
        )

        if df.empty or len(df) < 65:
            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )

        df = add_indicators(df)

        if len(df) < 65:
            return None

        x = df.iloc[-1]
        p = df.iloc[-2]

        price = float(x["Close"])

        ma5 = float(x["MA5"])
        ma10 = float(x["MA10"])
        ma20 = float(x["MA20"])
        ma60 = float(x["MA60"])

        ma20_prev = float(
            p["MA20"]
        )

        rsi = float(
            x["RSI14"]
        )

        macd = float(
            x["MACD"]
        )

        macd_signal = float(
            x["MACD_SIGNAL"]
        )

        macd_hist = float(
            x["MACD_HIST"]
        )

        macd_hist_prev = float(
            p["MACD_HIST"]
        )

        high20 = float(
            x["HIGH20_PREV"]
        )

        values = [
            price,
            ma5,
            ma10,
            ma20,
            ma60,
            rsi,
            macd,
            macd_signal,
            macd_hist,
            macd_hist_prev,
            high20,
        ]

        if not np.isfinite(
            values
        ).all():
            return None

        # --------------------------------------------------
        # Liquidity
        # --------------------------------------------------

        avg_value = float(
            x["VOL20"]
            * ma20
        )

        if not is_etf:

            if price < MIN_PRICE:
                return None

            if (
                avg_value
                < MIN_AVG_DAILY_VALUE
            ):
                return None

        # --------------------------------------------------
        # Original V3 conditions
        # --------------------------------------------------

        above_3ma = bool(
            price > ma5
            and price > ma10
            and price > ma20
        )

        trend = bool(
            price > ma20 > ma60
            and ma20 > ma20_prev
        )

        rsi_ok = bool(
            50 <= rsi <= 70
        )

        macd_strong = bool(
            macd > macd_signal
            and
            macd_hist
            > macd_hist_prev
        )

        breakout = bool(
            price > high20
        )

        if (
            pd.notna(
                x["VOL20"]
            )
            and float(
                x["VOL20"]
            ) > 0
        ):

            volume_ratio = float(
                x["Volume"]
                / x["VOL20"]
            )

        else:
            volume_ratio = 0.0

        volume_ok = bool(
            volume_ratio >= 1.5
        )

        checks = {

            "站上三短均":
                above_3ma,

            "趨勢多頭":
                trend,

            "RSI 50–70":
                rsi_ok,

            "MACD 動能增強":
                macd_strong,

            "突破前20日高點":
                breakout,

            "成交量 > 20日均量×1.5":
                volume_ok,
        }

        score = sum(
            checks.values()
        )

        if score < MIN_SCORE:
            return None

        reasons = [
            key
            for key, value
            in checks.items()
            if value
        ]

        # --------------------------------------------------
        # Original category
        # --------------------------------------------------

        if (
            breakout
            and volume_ok
            and above_3ma
        ):

            category = "強勢突破"

        elif (
            trend
            and above_3ma
        ):

            category = "多頭趨勢"

        else:

            category = "觀察名單"

        # --------------------------------------------------
        # V4 scores
        # --------------------------------------------------

        launch_score, launch_reasons = (
            calculate_launch_score(
                price,
                ma5,
                ma10,
                ma20,
                ma60,
                ma20_prev,
                rsi,
                macd,
                macd_signal,
                macd_hist,
                macd_hist_prev,
                high20,
                volume_ratio,
            )
        )

        confidence = (
            calculate_confidence(
                price,
                ma5,
                ma10,
                ma20,
                ma60,
                rsi,
                macd,
                macd_signal,
                macd_hist,
                macd_hist_prev,
                volume_ratio,
                high20,
            )
        )

        stars = confidence_stars(
            confidence
        )

        distance_ma20_pct = (
            (
                price - ma20
            )
            / ma20
            * 100
        )

        signal, overheated = (
            determine_signal(
                launch_score,
                confidence,
                rsi,
                distance_ma20_pct,
                price,
                high20,
            )
        )

        # --------------------------------------------------
        # ATR / price references
        # --------------------------------------------------

        atr = (
            float(
                x["ATR14"]
            )
            if pd.notna(
                x["ATR14"]
            )
            else np.nan
        )

        risk_reference = (
            round(
                price - 2 * atr,
                2
            )
            if np.isfinite(
                atr
            )
            else None
        )

        breakout_price = round(
            high20,
            2
        )

        # Entry reference zone
        #
        # Lower area is based around MA5/MA10.
        # This is a technical reference range,
        # not a guaranteed execution price.

        entry_low = round(
            max(
                ma10,
                ma20
            ),
            2
        )

        entry_high = round(
            min(
                price,
                high20
            ),
            2
        )

        if entry_high < entry_low:
            entry_high = round(
                price,
                2
            )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        return {

            "code":
                code,

            "name":
                name,

            "market":
                market,

            "ticker":
                ticker,

            "groups":
                groups,

            "is_etf":
                is_etf,

            "date":
                pd.Timestamp(
                    df.index[-1]
                ).strftime(
                    "%Y-%m-%d"
                ),

            "price":
                round(
                    price,
                    2
                ),

            # V3
            "score":
                int(score),

            "category":
                category,

            "above_3ma":
                above_3ma,

            "rsi":
                round(
                    rsi,
                    1
                ),

            "volume_ratio":
                round(
                    volume_ratio,
                    2
                ),

            "ma5":
                round(
                    ma5,
                    2
                ),

            "ma10":
                round(
                    ma10,
                    2
                ),

            "ma20":
                round(
                    ma20,
                    2
                ),

            "ma60":
                round(
                    ma60,
                    2
                ),

            "atr14":
                (
                    round(
                        atr,
                        2
                    )
                    if np.isfinite(
                        atr
                    )
                    else None
                ),

            "risk_reference":
                risk_reference,

            "reasons":
                reasons,

            # V4
            "launch_score":
                launch_score,

            "confidence":
                confidence,

            "confidence_stars":
                stars,

            "signal":
                signal,

            "overheated":
                overheated,

            "distance_ma20_pct":
                round(
                    distance_ma20_pct,
                    2
                ),

            "breakout_price":
                breakout_price,

            "entry_low":
                entry_low,

            "entry_high":
                entry_high,

            "launch_reasons":
                launch_reasons,

            # K-line data
            "chart":
                build_chart_data(
                    df
                ),
        }

    except Exception as exc:

        print(
            f"Skip {ticker}: "
            f"{str(exc)[:200]}"
        )

        return None


# ==========================================================
# Main
# ==========================================================

def main():

    universe = get_universe()

    print(
        f"V4 Universe: "
        f"{len(universe)} symbols"
    )

    rows = []

    for i, item in enumerate(
        universe,
        1
    ):

        result = scan_one(
            *item
        )

        if result:
            rows.append(
                result
            )

        print(
            f"{i}/"
            f"{len(universe)} "
            f"{item[1]}"
        )

        time.sleep(
            0.03
        )

    # ------------------------------------------------------
    # V4 ranking
    # ------------------------------------------------------

    rows.sort(
        key=lambda x: (
            x["launch_score"],
            x["confidence"],
            x["score"],
            x["volume_ratio"],
        ),
        reverse=True,
    )

    rows = rows[
        :MAX_RESULTS
    ]

    # ------------------------------------------------------
    # Group statistics
    # ------------------------------------------------------

    group_counts = {
        "半導體": 0,
        "AI": 0,
        "科技": 0,
        "ETF": 0,
    }

    signal_counts = {
        "符合進場條件": 0,
        "等待突破確認": 0,
        "過熱／不追價": 0,
        "條件不足": 0,
    }

    for row in rows:

        for group in row[
            "groups"
        ]:

            if group in group_counts:
                group_counts[
                    group
                ] += 1

        signal = row[
            "signal"
        ]

        if signal in signal_counts:
            signal_counts[
                signal
            ] += 1

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    now = datetime.now(
        TZ
    )

    payload = {

        "version":
            "V4",

        "updated":
            now.strftime(
                "%Y-%m-%d %H:%M"
            ),

        "timezone":
            "Asia/Taipei",

        "universe_count":
            len(universe),

        "strategy":
            (
                "V4 Launch Radar: "
                "MA structure + RSI + "
                "MACD momentum + "
                "20D breakout readiness + "
                "volume expansion + "
                "overheat control"
            ),

        "count":
            len(rows),

        "group_counts":
            group_counts,

        "signal_counts":
            signal_counts,

        "stocks":
            rows,
    }

    os.makedirs(
        "docs",
        exist_ok=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        "docs/data.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    # CSV does not need the full 60-day
    # chart array.
    csv_rows = []

    for row in rows:

        csv_row = dict(
            row
        )

        csv_row.pop(
            "chart",
            None
        )

        csv_rows.append(
            csv_row
        )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        "data/signals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "=============================="
    )
    print(
        "Taiwan Stock Scanner V4"
    )
    print(
        "=============================="
    )

    print(
        "Universe:",
        len(universe)
    )

    print(
        "Selected:",
        len(rows)
    )

    print(
        "Groups:",
        group_counts
    )

    print(
        "Signals:",
        signal_counts
    )


if __name__ == "__main__":
    main()
