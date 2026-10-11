"""V6.8 bullish-gap exit, causal daily-bar implementation.

Gap: today's low > yesterday's high.
Filled: a later day's low <= yesterday's high (gap bottom).
Swing low: pivot low with 3 bars on either side, known only after
the three confirmation bars close. Only pivots confirmed before the
gap are eligible, and the stop reference is frozen at gap formation.
Signals are evaluated at the close; execution belongs to the next bar.
"""
import math


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def gap_exit_series(df, pivot_bars=3):
    """Return one state dict per completed daily bar; no future reads."""
    bars = df.reset_index(drop=True)
    out = []
    confirmed_lows = []
    gap = None
    for i in range(len(bars)):
        high = _num(bars.iloc[i]["High"])
        low = _num(bars.iloc[i]["Low"])
        close = _num(bars.iloc[i]["Close"])
        if None in (high, low, close):
            raise ValueError("Missing OHLC value")
        # A pivot at i-pivot_bars becomes knowable only today.
        p = i - pivot_bars
        if p >= pivot_bars:
            window = [_num(bars.iloc[j]["Low"]) for j in range(p-pivot_bars, i+1)]
            if all(v is not None for v in window) and window[pivot_bars] == min(window):
                if window.count(window[pivot_bars]) == 1:
                    confirmed_lows.append((p, window[pivot_bars]))
        if i > 0:
            previous_high = _num(bars.iloc[i-1]["High"])
            if previous_high is not None and low > previous_high:
                # The latest gap supersedes the older one, including a filled gap.
                gap = {
                    "formed_index": i,
                    "bottom": previous_high,
                    "top": low,
                    "swing_low": confirmed_lows[-1][1] if confirmed_lows else None,
                    "filled": False,
                }
        if gap is not None and i > gap["formed_index"] and low <= gap["bottom"]:
            gap["filled"] = True
        exit_signal = bool(
            gap is not None and gap["filled"]
            and gap["swing_low"] is not None and close < gap["swing_low"]
        )
        out.append({
            "gap_formed_index": gap["formed_index"] if gap else None,
            "gap_bottom": gap["bottom"] if gap else None,
            "gap_top": gap["top"] if gap else None,
            "gap_filled": bool(gap and gap["filled"]),
            "reference_swing_low": gap["swing_low"] if gap else None,
            "exit_signal": exit_signal,
        })
    return out


def gap_exit_status(df):
    return gap_exit_series(df)[-1] if len(df) else {
        "gap_formed_index": None, "gap_bottom": None, "gap_top": None,
        "gap_filled": False, "reference_swing_low": None, "exit_signal": False,
    }
