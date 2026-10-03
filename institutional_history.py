"""Point-in-time T86 inputs for next-session-open research orders."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

FOREIGN = "外陸資買賣超股數(不含外資自營商)"
TRUST = "投信買賣超股數"


def parse_market_month(payload, month):
    month = pd.Timestamp(month).to_period("M")
    expected = f"{month.year - 1911:03d}年{month.month:02d}月"
    fields = payload.get("fields", [])
    if (payload.get("stat") != "OK" or expected not in payload.get("title", "")
            or "日期" not in fields or not payload.get("data")):
        raise ValueError(f"Invalid TWSE market calendar: {month}")
    dates = []
    for row in payload["data"]:
        year, m, day = map(int, row[fields.index("日期")].split("/"))
        date = pd.Timestamp(year=year + 1911, month=m, day=day)
        if date.to_period("M") != month:
            raise ValueError(f"Market calendar month mismatch: {date}")
        dates.append(date)
    index = pd.DatetimeIndex(dates)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Market calendar dates must be sorted and unique")
    return index


def load_market_calendar(start_date, end_date, cache_dir=".cache/t86/v1/market"):
    """Use actual monthly TWSE turnover dates, including unscheduled closures."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    months = pd.period_range(start, end, freq="M")
    dates = []
    for month in months:
        path = Path(cache_dir) / f"{month}.json"
        payload = None
        # The current month is incomplete and must always be refreshed.
        if month < end.to_period("M") and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                parse_market_month(cached, month.start_time)
                payload = cached
            except (ValueError, KeyError, TypeError):
                pass
        if payload is None:
            query = urllib.parse.urlencode({"response": "json", "date": month.start_time.strftime("%Y%m%d")})
            request = urllib.request.Request("https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?" + query,
                                             headers={"User-Agent": "Mozilla/5.0"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=25) as response:
                        payload = json.loads(response.read().decode("utf-8-sig"))
                    parse_market_month(payload, month.start_time)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
        dates.extend(parse_market_month(payload, month.start_time))
    index = pd.DatetimeIndex(dates)
    return index[(index >= start) & (index <= end)]


def filter_market_sessions(df, calendar):
    normalized = df.index.tz_localize(None).normalize()
    return df.loc[normalized.isin(calendar)].copy()


def parse_t86(payload, date):
    expected = f"{date.year - 1911:03d}年{date.month:02d}月{date.day:02d}日"
    if expected not in payload.get("title", ""):
        raise ValueError(f"T86 response date mismatch for {date.date()}")
    fields = payload.get("fields", [])
    required = ["證券代號", FOREIGN, TRUST]
    if payload.get("stat") != "OK" or not payload.get("data") or any(f not in fields for f in required):
        raise ValueError(f"T86 report unavailable or schema changed: {date.date()}")
    indices = [fields.index(f) for f in required]
    result = {}
    for row in payload["data"]:
        code = str(row[indices[0]]).strip()
        if code.isdigit() and len(code) == 4 and not code.startswith("0"):
            result[code] = [int(str(row[i]).replace(",", "").strip()) for i in indices[1:]]
    if not result:
        raise ValueError("Empty T86 common-stock report")
    return result


def fetch_daily(date, cache_dir):
    date = pd.Timestamp(date)
    path = Path(cache_dir) / f"{date:%Y-%m-%d}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("date") == str(date.date()) and cached.get("source") == "TWSE T86 v1":
            return cached["stocks"]
    query = urllib.parse.urlencode({"response": "json", "date": date.strftime("%Y%m%d"), "selectType": "ALLBUT0999"})
    request = urllib.request.Request("https://www.twse.com.tw/rwd/zh/fund/T86?" + query,
                                     headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                stocks = parse_t86(json.loads(response.read().decode("utf-8-sig")), date)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"date": str(date.date()), "source": "TWSE T86 v1", "stocks": stocks}, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.4)
            return stocks
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def load_reports(calendar, start_date, cache_dir=".cache/t86/v1"):
    # Fetch one actual prior trading session so January's first signal can
    # use two consecutive sessions. Never treat a missing session as a holiday.
    calendar = pd.DatetimeIndex(calendar)
    calendar = calendar.tz_localize(None).normalize().unique().sort_values()
    start = pd.Timestamp(start_date)
    prior = calendar[calendar < start]
    dates = calendar[calendar >= start]
    if len(prior):
        dates = prior[-1:].append(dates)
    reports, failures = {}, {}
    for date in dates:
        key = str(date.date())
        try:
            reports[key] = fetch_daily(date, cache_dir)
            print(f"[T86 OK] {key}", flush=True)
        except Exception as exc:
            failures[key] = str(exc)
            print(f"[T86 MISSING] {key}: {exc}", flush=True)
    return dates, reports, failures


def attach_gate(df, code, calendar, reports):
    daily = pd.DataFrame(index=calendar, columns=["Foreign", "Trust"], dtype=float)
    for date in calendar:
        values = reports.get(str(date.date()), {}).get(code)
        if values is not None:
            daily.loc[date] = values
    buys = daily.gt(0).all(axis=1)
    gate = buys & buys.shift(1, fill_value=False)
    normalized = df.index.tz_localize(None).normalize()
    out = df.copy()
    out["INSTITUTIONAL_2D"] = gate.reindex(normalized, fill_value=False).to_numpy()
    out["INSTITUTIONAL_AVAILABLE"] = daily.notna().all(axis=1).reindex(normalized, fill_value=False).to_numpy()
    return out
