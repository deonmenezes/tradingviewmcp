"""Mechanical building blocks: FVG, IFVG, CISD, liquidity sweep, BPR, SMT.

Each function operates on plain OHLC rows/DataFrames so it can be unit
tested with small hand-built fixtures (see tests/) and reused by detector.py
without re-deriving the same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Direction = Literal["bullish", "bearish"]


@dataclass
class FVG:
    """A 3-candle fair value gap imbalance."""

    direction: Direction
    timeframe: str
    start_time: pd.Timestamp  # timestamp of candle1
    end_time: pd.Timestamp  # timestamp of candle3
    top: float
    bottom: float
    mitigated: bool = False
    mitigated_at: pd.Timestamp | None = None

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


def detect_fvgs(df: pd.DataFrame, timeframe: str) -> list[FVG]:
    """Detect all 3-candle FVGs in a DataFrame of OHLC bars.

    Bullish FVG: gap between candle1.high and candle3.low when candle2
    displaces up (candle3.low > candle1.high).
    Bearish FVG: gap between candle1.low and candle3.high when candle2
    displaces down (candle3.high < candle1.low).
    """
    fvgs: list[FVG] = []
    if len(df) < 3:
        return fvgs

    idx = df.index
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(2, len(df)):
        c1_high, c1_low = highs[i - 2], lows[i - 2]
        c3_high, c3_low = highs[i], lows[i]

        if c3_low > c1_high:
            fvgs.append(
                FVG(
                    direction="bullish",
                    timeframe=timeframe,
                    start_time=idx[i - 2],
                    end_time=idx[i],
                    top=float(c3_low),
                    bottom=float(c1_high),
                )
            )
        elif c3_high < c1_low:
            fvgs.append(
                FVG(
                    direction="bearish",
                    timeframe=timeframe,
                    start_time=idx[i - 2],
                    end_time=idx[i],
                    top=float(c1_low),
                    bottom=float(c3_high),
                )
            )
    return fvgs


def mark_mitigations(fvgs: list[FVG], df: pd.DataFrame) -> None:
    """Mutates each FVG's ``mitigated``/``mitigated_at`` in place: price has
    traded back through the gap after it formed.
    """
    idx = df.index
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for fvg in fvgs:
        bars_after = idx.searchsorted(fvg.end_time, side="right")
        for j in range(bars_after, len(df)):
            if lows[j] <= fvg.top and highs[j] >= fvg.bottom:
                fvg.mitigated = True
                fvg.mitigated_at = idx[j]
                break


def unmitigated_fvgs(fvgs: list[FVG], direction: Direction | None = None) -> list[FVG]:
    out = [f for f in fvgs if not f.mitigated]
    if direction is not None:
        out = [f for f in out if f.direction == direction]
    return out


def closest_unmitigated_fvg(
    fvgs: list[FVG], price: float, direction: Direction | None = None
) -> FVG | None:
    """The unmitigated FVG whose midpoint is closest to ``price``."""
    candidates = unmitigated_fvgs(fvgs, direction)
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f.midpoint - price))


def detect_bpr(bull_fvgs: list[FVG], bear_fvgs: list[FVG]) -> list[tuple[FVG, FVG]]:
    """Balanced Price Range: an opposing FVG pair that overlap, where the
    later one inverses the earlier (forms immediately after). Treated as a
    high-quality POI per config.
    """
    pairs: list[tuple[FVG, FVG]] = []
    for a in bull_fvgs:
        for b in bear_fvgs:
            earlier, later = (a, b) if a.end_time <= b.end_time else (b, a)
            overlap = min(earlier.top, later.top) - max(earlier.bottom, later.bottom)
            if overlap > 0:
                pairs.append((earlier, later))
    return pairs


def body_closes_through(candle: pd.Series, level: float, direction: Direction) -> bool:
    """Body closure check used by IFVG/CISD — a wick trading through the
    level does not count; the candle's CLOSE must be beyond it.
    """
    if direction == "bullish":
        return bool(candle["close"] > level)
    else:
        return bool(candle["close"] < level)


def detect_ifvg(
    df: pd.DataFrame, opposing_fvg: FVG, entry_direction: Direction
) -> pd.Timestamp | None:
    """Tier 1 — Inversion FVG: first candle (after the opposing FVG formed)
    whose BODY closes back through it. For a long, body closes above the
    most recent bearish FVG (and vice versa for shorts).
    """
    after = df[df.index > opposing_fvg.end_time]
    level = opposing_fvg.top if entry_direction == "bullish" else opposing_fvg.bottom
    for ts, candle in after.iterrows():
        if body_closes_through(candle, level, entry_direction):
            return ts
    return None


@dataclass
class CISDRun:
    direction: Direction  # direction of the manipulation run being broken
    first_candle_time: pd.Timestamp
    first_candle_open: float
    run_length: int


def find_last_consecutive_run(df: pd.DataFrame, direction: Direction) -> CISDRun | None:
    """Last consecutive run of candles in ``direction`` (down for bearish
    manipulation feeding a long CISD, up for bullish manipulation feeding a
    short CISD). Returns the open of the first candle of that run.
    """
    is_down = df["close"] < df["open"]
    is_up = df["close"] > df["open"]
    mask = is_down if direction == "bearish" else is_up

    if len(df) == 0 or not bool(mask.iloc[-1]):
        # walk back from the end to find the most recent finished run
        run_end = None
        for i in range(len(mask) - 1, -1, -1):
            if mask.iloc[i]:
                run_end = i
                break
        if run_end is None:
            return None
    else:
        run_end = len(mask) - 1

    run_start = run_end
    while run_start - 1 >= 0 and mask.iloc[run_start - 1]:
        run_start -= 1

    run_length = run_end - run_start + 1
    first = df.iloc[run_start]
    return CISDRun(
        direction=direction,
        first_candle_time=df.index[run_start],
        first_candle_open=float(first["open"]),
        run_length=run_length,
    )


def detect_cisd(
    df: pd.DataFrame,
    manipulation_direction: Direction,
    allow_single_candle: bool = False,
) -> pd.Timestamp | None:
    """Tier 2 — Change in State of Delivery.

    manipulation_direction="bearish" means the manipulation leg ran down
    (setting up a long); we look for a body close above the run's first
    candle open. Mirror for "bullish" manipulation -> short CISD.
    """
    run = find_last_consecutive_run(df, manipulation_direction)
    if run is None:
        return None
    if run.run_length < 2 and not allow_single_candle:
        return None

    after = df[df.index > run.first_candle_time]
    entry_direction: Direction = "bullish" if manipulation_direction == "bearish" else "bearish"
    for ts, candle in after.iterrows():
        if entry_direction == "bullish" and candle["close"] > run.first_candle_open:
            return ts
        if entry_direction == "bearish" and candle["close"] < run.first_candle_open:
            return ts
    return None


def detect_liquidity_sweep(
    df: pd.DataFrame, level: float, direction: Direction
) -> pd.Timestamp | None:
    """A sweep of LRL/trendline liquidity (e.g. the accumulation high/low):
    a wick trades beyond ``level`` then closes back on the other side.

    direction="bullish" => sweep of a low (wick below level, close above).
    direction="bearish" => sweep of a high (wick above level, close below).
    """
    for ts, candle in df.iterrows():
        if direction == "bullish" and candle["low"] < level and candle["close"] > level:
            return ts
        if direction == "bearish" and candle["high"] > level and candle["close"] < level:
            return ts
    return None


def smt_divergence(
    primary: pd.DataFrame, correlated: pd.DataFrame, at: Literal["high", "low"]
) -> bool:
    """SMT divergence between two correlated assets (e.g. NQ vs ES) over
    the given aligned window of bars.

    at="high": bearish SMT — primary makes a higher high while correlated
    makes a lower high (or fails to sweep its prior high).
    at="low": bullish SMT — primary makes a lower low while correlated
    makes a higher low.
    """
    if len(primary) < 2 or len(correlated) < 2:
        return False

    if at == "high":
        primary_higher_high = primary["high"].iloc[-1] > primary["high"].iloc[:-1].max()
        correlated_lower_high = correlated["high"].iloc[-1] < correlated["high"].iloc[:-1].max()
        return bool(primary_higher_high and correlated_lower_high)
    else:
        primary_lower_low = primary["low"].iloc[-1] < primary["low"].iloc[:-1].min()
        correlated_higher_low = correlated["low"].iloc[-1] > correlated["low"].iloc[:-1].min()
        return bool(primary_lower_low and correlated_higher_low)
