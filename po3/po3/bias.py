"""Multi-timeframe directional bias from PO3 profile + OHLC genetic makeup.

A candle's "genetic makeup" is the order in which its OHLC points form:
bullish candles go Open -> Low -> High -> Close; bearish candles go
Open -> High -> Low -> Close. We approximate which wick formed first using
intrabar data when available, falling back to a close-vs-open heuristic
when it is not (documented inline — this is an approximation, not a claim
of certainty about intrabar sequencing from OHLC-only data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Bias = Literal["long", "short", "mixed"]


@dataclass
class BiasResult:
    bias: Bias
    clarity: float  # 0..1, higher = more confident / unanimous across timeframes
    components: dict[str, Bias]


def candle_bias(candle: pd.Series) -> Bias:
    """OHLC-only approximation of which wick formed first.

    True intrabar sequencing requires tick/lower-TF data; from OHLC alone we
    use the standard PO3 heuristic: a candle that closes in the top half of
    its range is treated as having swept the low first (bullish genetic
    makeup), and vice versa for the bottom half.
    """
    rng = candle["high"] - candle["low"]
    if rng <= 0:
        return "mixed"
    close_position = (candle["close"] - candle["low"]) / rng
    if close_position >= 0.5:
        return "long"
    return "short"


def daily_po3_bias(daily_bars: pd.DataFrame, as_of: pd.Timestamp) -> Bias:
    """Bias from the most recently completed daily candle's genetic makeup."""
    prior = daily_bars[daily_bars.index < as_of]
    if prior.empty:
        return "mixed"
    return candle_bias(prior.iloc[-1])


def weekly_po3_bias(weekly_bars: pd.DataFrame, as_of: pd.Timestamp) -> Bias:
    prior = weekly_bars[weekly_bars.index < as_of]
    if prior.empty:
        return "mixed"
    return candle_bias(prior.iloc[-1])


def thirty_min_po3_bias(bars_30m: pd.DataFrame, as_of: pd.Timestamp) -> Bias:
    """30-min PO3: direction implied by the open of the most recent
    completed 30-minute candle relative to its close so far.
    """
    prior = bars_30m[bars_30m.index < as_of]
    if prior.empty:
        return "mixed"
    candle = prior.iloc[-1]
    if candle["close"] > candle["open"]:
        return "long"
    if candle["close"] < candle["open"]:
        return "short"
    return "mixed"


def resolve_bias(
    daily_bars: pd.DataFrame,
    as_of: pd.Timestamp,
    weekly_bars: pd.DataFrame | None = None,
    bars_30m: pd.DataFrame | None = None,
) -> BiasResult:
    """Combine Daily (required) + Weekly/30m (optional confirmation) into a
    single bias with a clarity score in [0, 1] based on agreement across
    the timeframes that were supplied.
    """
    components: dict[str, Bias] = {"daily": daily_po3_bias(daily_bars, as_of)}
    if weekly_bars is not None:
        components["weekly"] = weekly_po3_bias(weekly_bars, as_of)
    if bars_30m is not None:
        components["30min"] = thirty_min_po3_bias(bars_30m, as_of)

    votes = [v for v in components.values() if v != "mixed"]
    if not votes:
        return BiasResult(bias="mixed", clarity=0.0, components=components)

    longs = votes.count("long")
    shorts = votes.count("short")
    total = len(votes)

    if longs == total:
        return BiasResult(bias="long", clarity=1.0, components=components)
    if shorts == total:
        return BiasResult(bias="short", clarity=1.0, components=components)

    majority_bias: Bias = "long" if longs > shorts else "short"
    clarity = max(longs, shorts) / total
    if longs == shorts:
        return BiasResult(bias="mixed", clarity=0.5, components=components)
    return BiasResult(bias=majority_bias, clarity=clarity, components=components)
