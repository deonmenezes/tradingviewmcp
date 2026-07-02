"""Performance metrics and splits over a trade log."""

from __future__ import annotations

import pandas as pd


def _expectancy(df: pd.DataFrame) -> float:
    return float(df["r_achieved"].mean()) if len(df) else 0.0


def _profit_factor(df: pd.DataFrame) -> float:
    gains = df.loc[df["r_achieved"] > 0, "r_achieved"].sum()
    losses = -df.loc[df["r_achieved"] < 0, "r_achieved"].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _max_consecutive_losses(df: pd.DataFrame) -> int:
    streak = 0
    worst = 0
    for win in df.sort_values("session_open")["win"]:
        if not win:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def overall_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_r": None,
            "expectancy": None,
            "profit_factor": None,
            "max_consecutive_losses": 0,
            "trades_per_week": None,
        }

    weeks = max((df["session_open"].max() - df["session_open"].min()).days / 7.0, 1.0 / 7.0)

    return {
        "trades": int(len(df)),
        "win_rate": float(df["win"].mean()),
        "avg_r": float(df["r_achieved"].mean()),
        "expectancy": _expectancy(df),
        "profit_factor": _profit_factor(df),
        "max_consecutive_losses": _max_consecutive_losses(df),
        "trades_per_week": float(len(df) / weeks),
    }


def r_distribution(df: pd.DataFrame, bins: int = 10) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return pd.cut(df["r_achieved"], bins=bins).value_counts().sort_index()


def split_by_session(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    local_hour = df["session_open"].dt.tz_convert(tz).dt.hour
    return df.assign(session_hour=local_hour).groupby("session_hour").apply(
        overall_stats, include_groups=False
    ).apply(pd.Series)


def split_by_weekday(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    weekday = df["session_open"].dt.tz_convert(tz).dt.day_name()
    return df.assign(weekday=weekday).groupby("weekday").apply(
        overall_stats, include_groups=False
    ).apply(pd.Series)


def split_by_poi_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("poi_timeframe").apply(overall_stats, include_groups=False).apply(pd.Series)
