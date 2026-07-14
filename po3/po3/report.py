"""Outputs: trade log CSV, equity curve PNG, P&L calendar, --explain mode."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from po3.backtest import BacktestResult


def write_trade_log(result: BacktestResult, path: str) -> pd.DataFrame:
    df = result.trade_log_df()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def equity_curve(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if df.empty:
        ax.set_title("Equity curve (no trades)")
    else:
        ordered = df.sort_values("session_open")
        cumulative_r = ordered["r_achieved"].cumsum()
        ax.plot(ordered["session_open"], cumulative_r, marker="o", markersize=3)
        ax.set_title("Cumulative R")
        ax.set_xlabel("Session open")
        ax.set_ylabel("Cumulative R")
        ax.axhline(0, color="gray", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pnl_calendar(df: pd.DataFrame, path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Per-day P&L (in R) to inspect the 'one bad day wipes the month' risk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        daily = pd.DataFrame(columns=["date", "r_total"])
    else:
        local_date = df["session_open"].dt.tz_convert(tz).dt.date
        daily = df.assign(date=local_date).groupby("date")["r_achieved"].sum().reset_index(
            name="r_total"
        )
    daily.to_csv(path, index=False)
    return daily


def explain_sessions(setups, limit: int | None = None) -> str:
    chosen = setups if limit is None else setups[:limit]
    return "\n\n".join(s.explain() for s in chosen)
