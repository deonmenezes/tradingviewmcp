"""Data loading and session-aligned resampling.

Everything is derived from 1-minute bars. Higher timeframes (5m/15m/1H/4H/
Daily/Weekly) are NOT trusted if pre-aggregated upstream — they are always
rebuilt here from 1m using the configured session open, because every
timeframe boundary in this strategy depends on the 18:00-ET session open.
"""

from __future__ import annotations

import pandas as pd

from po3.config import SessionConfig

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


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
