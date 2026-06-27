"""Data loading and session-aligned resampling.

Everything is derived from 1-minute bars. Higher timeframes (5m/15m/1H/4H/
Daily/Weekly) are NOT trusted if pre-aggregated upstream — they are always
rebuilt here from 1m using the configured session open, because every
timeframe boundary in this strategy depends on the 18:00-ET session open.
"""

from __future__ import annotations

import warnings

import pandas as pd

from po3.config import SessionConfig

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when loaded data fails a load-time sanity check that must halt
    the run rather than silently produce an inaccurate backtest."""


def validate_strict(df: pd.DataFrame, label: str = "series") -> dict:
    """Hard validation gate (HALT on failure, per spec §4): duplicate or
    non-monotonic timestamps, non-positive prices/volume, and missing-minute
    gaps. Returns a dict of stats for the data-quality report; raises
    ``DataValidationError`` on anything that would silently corrupt the
    backtest rather than just costing some coverage.
    """
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()]
        raise DataValidationError(
            f"{label}: {len(dupes)} duplicate timestamp(s), e.g. {dupes[0]}. "
            "Refusing to backtest on data with duplicate bars."
        )
    if not df.index.is_monotonic_increasing:
        raise DataValidationError(f"{label}: timestamps are not strictly increasing.")

    bad_price = (df[["open", "high", "low", "close"]] <= 0).any(axis=None)
    if bad_price:
        raise DataValidationError(f"{label}: found zero/negative OHLC price(s).")
    if (df["volume"] < 0).any():
        raise DataValidationError(f"{label}: found negative volume.")

    expected_minutes = pd.date_range(df.index[0], df.index[-1], freq="1min")
    missing_pct = 100.0 * (1.0 - len(df.index) / len(expected_minutes)) if len(expected_minutes) else 0.0

    return {
        "label": label,
        "rows": len(df),
        "start": df.index[0],
        "end": df.index[-1],
        "missing_minutes_pct": missing_pct,
    }


def detect_roll_dates(
    df: pd.DataFrame, max_jump_pct: float = 8.0
) -> pd.DatetimeIndex:
    """Identify likely contract-roll dates from large close-to-close jumps,
    so they can be flagged/excluded per spec §3 rather than silently treated
    as real price action.
    """
    pct_change = df["close"].pct_change().abs() * 100.0
    flagged = df.index[pct_change > max_jump_pct]
    return pd.DatetimeIndex(sorted({ts.normalize() for ts in flagged}))


def exclude_roll_sessions(
    df: pd.DataFrame, roll_dates: pd.DatetimeIndex, buffer_hours: int = 24
) -> pd.DataFrame:
    """Drop bars within ``buffer_hours`` of a flagged roll date, since FVG/
    sweep detection can mistake the roll-induced gap for real manipulation.
    """
    if len(roll_dates) == 0:
        return df
    mask = pd.Series(True, index=df.index)
    buffer = pd.Timedelta(hours=buffer_hours)
    for rd in roll_dates:
        mask &= ~((df.index >= rd - buffer) & (df.index <= rd + buffer))
    return df[mask]


def data_quality_report(
    primary_stats: dict,
    correlated_stats: dict | None,
    roll_dates: pd.DatetimeIndex,
    alignment_pct: float | None,
) -> str:
    lines = ["=== Data Quality Report ==="]
    for stats in (primary_stats, correlated_stats):
        if stats is None:
            continue
        lines.append(
            f"{stats['label']}: {stats['rows']} rows, {stats['start']} -> {stats['end']}, "
            f"missing minutes: {stats['missing_minutes_pct']:.2f}%"
        )
    lines.append(f"Roll dates flagged: {len(roll_dates)}" + (f" ({list(roll_dates)})" if len(roll_dates) else ""))
    if alignment_pct is not None:
        lines.append(f"NQ/ES minute alignment: {alignment_pct:.2f}%")
    return "\n".join(lines)


def validate_alignment(primary: pd.DataFrame, correlated: pd.DataFrame, max_misaligned_pct: float = 1.0) -> None:
    """Warn (not raise) if primary/correlated 1m bars aren't minute-aligned.

    SMT divergence compares bar-for-bar; silently misaligned timestamps would
    quietly compare the wrong candles. ``max_misaligned_pct`` is the percent
    of primary bars allowed to have no matching correlated timestamp before
    we warn (a few missing bars from feed gaps are normal).
    """
    shared = primary.index.intersection(correlated.index)
    if len(primary) == 0:
        return
    missing_pct = 100.0 * (1.0 - len(shared) / len(primary))
    if missing_pct > max_misaligned_pct:
        warnings.warn(
            f"primary/correlated 1m bars are only {100 - missing_pct:.1f}% "
            f"minute-aligned ({missing_pct:.1f}% of primary timestamps have no "
            "matching correlated bar) — SMT divergence tags may be comparing "
            "misaligned candles. Check feed alignment before trusting SMT output.",
            stacklevel=2,
        )


def detect_abnormal_jumps(
    df: pd.DataFrame, max_jump_pct: float = 8.0, label: str = "series"
) -> pd.DataFrame:
    """Flag bar-to-bar close jumps larger than ``max_jump_pct`` — a common
    signature of contract-roll stitching artifacts (back-adjusted/continuous
    data splicing in a price gap). Returns the flagged rows; only warns, does
    not modify or drop data, since a real large move is also possible.
    """
    pct_change = df["close"].pct_change().abs() * 100.0
    flagged = df[pct_change > max_jump_pct]
    if not flagged.empty:
        warnings.warn(
            f"{label}: {len(flagged)} bar(s) with >{max_jump_pct}% close-to-close "
            "jump detected — possible contract-roll stitching artifact (false "
            f"wick/gap). First flagged timestamp: {flagged.index[0]}. Inspect "
            "before trusting backtest results around these bars.",
            stacklevel=2,
        )
    return flagged


def load_ohlcv_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """Load a single-timeframe OHLCV CSV into a tz-aware, sorted DataFrame.

    The CSV is expected to have a ``timestamp`` column plus the columns in
    ``REQUIRED_COLUMNS``. The timestamp is interpreted as ``tz`` if it is
    naive, then converted to UTC for internal storage.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    df.index = df.index.tz_convert("UTC")
    return df[list(REQUIRED_COLUMNS)]


def _session_anchor(session: SessionConfig) -> str:
    """Pandas resample 'origin' anchor expressed as an offset alias.

    We resample in the session timezone so that day/week boundaries land on
    the configured session open hour rather than midnight UTC.
    """
    return f"{session.session_open_hour}h"


def resample_from_1m(
    df_1m: pd.DataFrame,
    rule: str,
    session: SessionConfig,
) -> pd.DataFrame:
    """Resample 1-minute bars to a higher timeframe, with bin boundaries
    anchored to the session open (so e.g. '4h' bins open at 18:00, 22:00,
    02:00, 06:00, 10:00, 14:00 ET, not at UTC midnight).
    """
    local = df_1m.tz_convert(session.timezone)

    offset = pd.Timedelta(hours=session.session_open_hour % 24)
    shifted = local.copy()
    shifted.index = shifted.index - offset

    agg = shifted.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg.index = agg.index + offset
    return agg.tz_convert("UTC")


def four_hour_bars(df_1m: pd.DataFrame, session: SessionConfig) -> pd.DataFrame:
    """Build 4H bars opening exactly at the configured ``four_hour_opens``."""
    bars = resample_from_1m(df_1m, "4h", session)
    local_hours = bars.index.tz_convert(session.timezone).hour
    mask = local_hours.isin(session.four_hour_opens)
    return bars[mask]


def daily_bars(df_1m: pd.DataFrame, session: SessionConfig) -> pd.DataFrame:
    return resample_from_1m(df_1m, "1D", session)


def weekly_bars(df_1m: pd.DataFrame, session: SessionConfig) -> pd.DataFrame:
    return resample_from_1m(df_1m, "1W", session)


def minute_bars(df_1m: pd.DataFrame, minutes: int, session: SessionConfig) -> pd.DataFrame:
    return resample_from_1m(df_1m, f"{minutes}min", session)
