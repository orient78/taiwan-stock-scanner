import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import twstock
import yfinance as yf

TZ = ZoneInfo("Asia/Taipei")

MIN_PRICE = 10
MIN_AVG_DAILY_VALUE = 20_000_000
MAX_RESULTS = 30
MIN_SCORE = 3


def get_universe():
    stocks = []

    for code, info in twstock.codes.items():
        if not (
            code.isdigit()
            and len(code) == 4
            and info.type == "股票"
        ):
            continue

        if info.market == "上市":
            suffix = ".TW"
        elif info.market == "上櫃":
            suffix = ".TWO"
        else:
            continue

        stocks.append(
            (
                code + suffix,
                code,
                info.name,
                info.market,
            )
        )

    return stocks


def add_indicators(df):

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    # Short-term moving averages
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()

    # Medium-term trend
    df["MA60"] = close.rolling(60).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI14"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False,
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

    # ATR
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
            pd.MultiIndex,
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

        avg_value = float(
            x["VOL20"] * ma20
        )

        if (
            not np.isfinite(
                [
                    price,
                    ma5,
                    ma10,
                    ma20,
                    ma60,
                    float(x["RSI14"]),
                ]
            ).all()
            or price < MIN_PRICE
            or avg_value
            < MIN_AVG_DAILY_VALUE
        ):
            return None

        # --------------------------------
        # 1. Above 3 short-term averages
        # --------------------------------

        above_3ma = bool(
            price > ma5
            and price > ma10
            and price > ma20
        )

        # --------------------------------
        # 2. Bull trend
        # --------------------------------

        trend = bool(
            price > ma20 > ma60
            and ma20 > float(
                p["MA20"]
            )
        )

        # --------------------------------
        # 3. RSI
        # --------------------------------

        rsi_ok = bool(
            50
            <= float(x["RSI14"])
            <= 70
        )

        # --------------------------------
        # 4. MACD momentum
        # --------------------------------

        macd_strong = bool(
            float(x["MACD"])
            > float(
                x["MACD_SIGNAL"]
            )
            and float(
                x["MACD_HIST"]
            )
            > float(
                p["MACD_HIST"]
            )
        )

        # --------------------------------
        # 5. Breakout
        # --------------------------------

        breakout = bool(
            price
            > float(
                x["HIGH20_PREV"]
            )
        )

        # --------------------------------
        # 6. Volume
        # --------------------------------

        if x["VOL20"]:
            volume_ratio = float(
                x["Volume"]
                / x["VOL20"]
            )
        else:
            volume_ratio = 0

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
            k
            for k, v
            in checks.items()
            if v
        ]

        # --------------------------------
        # Category
        # --------------------------------

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

        atr = (
            float(x["ATR14"])
            if pd.notna(
                x["ATR14"]
            )
            else np.nan
        )

        return {

            "code": code,
            "name": name,
            "market": market,
            "ticker": ticker,

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
                round(
                    float(
                        x["RSI14"]
                    ),
                    1,
                ),

            "volume_ratio":
                round(
                    volume_ratio,
                    2,
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
                round(atr, 2)
                if np.isfinite(atr)
                else None,

            "risk_reference":
                round(
                    price - 2 * atr,
                    2,
                )
                if np.isfinite(atr)
                else None,

            "reasons":
                reasons,
        }

    except Exception as exc:

        print(
            f"Skip {ticker}: "
            f"{str(exc)[:120]}"
        )

        return None


def main():

    universe = get_universe()

    print(
        f"Universe: "
        f"{len(universe)} stocks"
    )

    rows = []

    for i, item in enumerate(
        universe,
        1,
    ):

        result = scan_one(
            *item
        )

        if result:
            rows.append(
                result
            )

        if i % 100 == 0:
            print(
                f"{i}/"
                f"{len(universe)}"
            )

        time.sleep(0.03)

    # --------------------------------
    # Ranking
    # --------------------------------

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

    now = datetime.now(TZ)

    payload = {

        "updated":
            now.strftime(
                "%Y-%m-%d %H:%M"
            ),

        "timezone":
            "Asia/Taipei",

        "strategy":
            (
                "MA5/MA10/MA20 + "
                "MA20/MA60 + "
                "RSI14 + MACD + "
                "20D breakout + volume"
            ),

        "count":
            len(rows),

        "stocks":
            rows,
    }

    os.makedirs(
        "docs",
        exist_ok=True,
    )

    os.makedirs(
        "data",
        exist_ok=True,
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
            indent=2,
        )

    pd.DataFrame(
        rows
    ).to_csv(
        "data/signals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Selected: "
        f"{len(rows)}"
    )


if __name__ == "__main__":
    main()
