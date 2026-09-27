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


def get_universe():
    """
    V3:
    Only scan symbols defined in stock_pool.py.
    """

    stocks = []

    for code in get_stock_codes():

        info = twstock.codes.get(code)

        if info is None:
            print(f"Unknown code: {code}")
            continue

        # ETF
        is_etf = "ETF" in get_groups(code)

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
                get_groups(code),
                is_etf,
            )
        )

    return stocks


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

    df["MACD"] = (
        ema12 - ema26
    )

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

        rsi = float(x["RSI14"])

        if not np.isfinite(
            [
                price,
                ma5,
                ma10,
                ma20,
                ma60,
                rsi,
            ]
        ).all():
            return None

        # ----------------------------------
        # Liquidity filter
        # ----------------------------------

        avg_value = float(
            x["VOL20"] * ma20
        )

        # ETF 不使用個股價格 / 成交額硬篩選
        if not is_etf:

            if price < MIN_PRICE:
                return None

            if (
                avg_value
                < MIN_AVG_DAILY_VALUE
            ):
                return None

        # ----------------------------------
        # Technical conditions
        # ----------------------------------

        above_3ma = bool(
            price > ma5
            and price > ma10
            and price > ma20
        )

        trend = bool(
            price > ma20 > ma60
            and ma20
            > float(p["MA20"])
        )

        rsi_ok = bool(
            50 <= rsi <= 70
        )

        macd_strong = bool(
            float(x["MACD"])
            > float(
                x["MACD_SIGNAL"]
            )
            and
            float(
                x["MACD_HIST"]
            )
            > float(
                p["MACD_HIST"]
            )
        )

        breakout = bool(
            price
            > float(
                x["HIGH20_PREV"]
            )
        )

        if (
            pd.notna(x["VOL20"])
            and float(x["VOL20"]) > 0
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

        # ----------------------------------
        # Signal category
        # ----------------------------------

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

        # ----------------------------------
        # ATR risk reference
        # ----------------------------------

        atr = (
            float(x["ATR14"])
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
            if np.isfinite(atr)
            else None
        )

        return {

            "code":
                code,

            "name":
                name,

            "market":
                market,

            "ticker":
                ticker,

            # V3 industry groups
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
                round(price, 2),

            "score":
                int(score),

            "category":
                category,

            "above_3ma":
                above_3ma,

            "rsi":
                round(rsi, 1),

            "volume_ratio":
                round(
                    volume_ratio,
                    2
                ),

            "ma5":
                round(ma5, 2),

            "ma10":
                round(ma10, 2),

            "ma20":
                round(ma20, 2),

            "ma60":
                round(ma60, 2),

            "atr14":
                (
                    round(atr, 2)
                    if np.isfinite(atr)
                    else None
                ),

            "risk_reference":
                risk_reference,

            "reasons":
                reasons,
        }

    except Exception as exc:

        print(
            f"Skip {ticker}: "
            f"{str(exc)[:150]}"
        )

        return None


def main():

    universe = get_universe()

    print(
        f"V3 Universe: "
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

        # Small delay to reduce API pressure
        time.sleep(0.03)

    # ----------------------------------
    # Ranking
    # ----------------------------------

    rows.sort(
        key=lambda x: (
            x["score"],
            x["volume_ratio"],
        ),
        reverse=True,
    )

    rows = rows[
        :MAX_RESULTS
    ]

    # ----------------------------------
    # Group statistics
    # ----------------------------------

    group_counts = {
        "半導體": 0,
        "AI": 0,
        "科技": 0,
        "ETF": 0,
    }

    for row in rows:

        for group in row["groups"]:

            if group in group_counts:
                group_counts[group] += 1

    # ----------------------------------
    # Output
    # ----------------------------------

    now = datetime.now(TZ)

    payload = {

        "version":
            "V3",

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
                "Focused Tech/AI/"
                "Semiconductor/ETF + "
                "MA5/MA10/MA20 + "
                "MA20/MA60 + RSI14 + "
                "MACD + 20D breakout + "
                "volume"
            ),

        "count":
            len(rows),

        "group_counts":
            group_counts,

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

    pd.DataFrame(
        rows
    ).to_csv(
        "data/signals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("==============================")
    print("Taiwan Stock Scanner V3")
    print("==============================")

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


if __name__ == "__main__":
    main()
