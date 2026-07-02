"""Walk-forward backtest over 4H session opens: trade management + R accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from po3 import data as data_mod
from po3 import detector as det
from po3.config import Config


@dataclass
class TradeRecord:
    session_open: pd.Timestamp
    bias: str
    poi_type: str
    poi_timeframe: str
    confirmation_tier: str
    entry: float
    stop: float
    targets: list[float]
    exit_price: float
    exit_time: pd.Timestamp
    r_achieved: float
    win: bool
    mfe_r: float
    mae_r: float
    smt_present: bool | None = None


def _simulate_trade(
    setup: det.SetupResult, df_after: pd.DataFrame, cfg: Config
) -> TradeRecord | None:
    direction = setup.direction
    entry = setup.entry
    stop = setup.stop
    targets = setup.targets
    if direction is None or entry is None or stop is None or not targets:
        return None

    risk = abs(entry - stop)
    if risk == 0:
        return None

    remaining = 1.0
    realized_r = 0.0
    trim_fractions = list(cfg.risk.trim_fractions)
    trim_r = list(cfg.risk.trim_r_multiples)
    current_stop = stop
    moved_to_be = False

    mfe_r = 0.0
    mae_r = 0.0
    exit_price = entry
    exit_time = df_after.index[0] if len(df_after) else setup.confirmation_time

    for ts, candle in df_after.iterrows():
        if direction == "long":
            move_favor = (candle["high"] - entry) / risk
            move_against = (entry - candle["low"]) / risk
        else:
            move_favor = (entry - candle["low"]) / risk
            move_against = (candle["high"] - entry) / risk

        mfe_r = max(mfe_r, move_favor)
        mae_r = max(mae_r, move_against)

        stop_hit = (
            (direction == "long" and candle["low"] <= current_stop)
            or (direction == "short" and candle["high"] >= current_stop)
        )
        if stop_hit:
            stop_r = (current_stop - entry) / risk if direction == "long" else (entry - current_stop) / risk
            realized_r += remaining * stop_r
            exit_price = current_stop
            exit_time = ts
            remaining = 0.0
            break

        for i, r_level in enumerate(trim_r):
            if remaining <= 0:
                break
            level_price = entry + r_level * risk if direction == "long" else entry - r_level * risk
            hit = (direction == "long" and candle["high"] >= level_price) or (
                direction == "short" and candle["low"] <= level_price
            )
            if hit and trim_fractions[i] > 0:
                trimmed = min(trim_fractions[i], remaining)
                realized_r += trimmed * r_level
                remaining -= trimmed
                trim_fractions[i] = 0.0
                exit_price = level_price
                exit_time = ts

        if not moved_to_be and mfe_r >= cfg.risk.breakeven_r:
            current_stop = entry
            moved_to_be = True

        if remaining <= 0:
            break

    if remaining > 0:
        last = df_after.iloc[-1]
        last_price = last["close"]
        last_r = (last_price - entry) / risk if direction == "long" else (entry - last_price) / risk
        realized_r += remaining * last_r
        exit_price = last_price
        exit_time = df_after.index[-1]

    return TradeRecord(
        session_open=setup.session_open,
        bias=direction,
        poi_type="FVG" if setup.poi is not None else "sweep_fallback",
        poi_timeframe=setup.poi.timeframe if setup.poi is not None else "n/a",
        confirmation_tier=setup.confirmation_tier or "n/a",
        entry=entry,
        stop=stop,
        targets=targets,
        exit_price=float(exit_price),
        exit_time=exit_time,
        r_achieved=float(realized_r),
        win=realized_r > 0,
        mfe_r=float(mfe_r),
        mae_r=float(mae_r),
        smt_present=setup.smt_present,
    )


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    setups: list[det.SetupResult] = field(default_factory=list)

    def trade_log_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append(
                {
                    "timestamp": t.session_open,
                    "session_open": t.session_open,
                    "bias": t.bias,
                    "poi_type": t.poi_type,
                    "poi_timeframe": t.poi_timeframe,
                    "confirmation_tier": t.confirmation_tier,
                    "entry": t.entry,
                    "stop": t.stop,
                    "tp1": t.targets[0] if len(t.targets) > 0 else None,
                    "tp2": t.targets[1] if len(t.targets) > 1 else None,
                    "tp3": t.targets[2] if len(t.targets) > 2 else None,
                    "exit": t.exit_price,
                    "exit_time": t.exit_time,
                    "r_achieved": t.r_achieved,
                    "win": t.win,
                    "mfe_r": t.mfe_r,
                    "mae_r": t.mae_r,
                    "smt_present": t.smt_present,
                }
            )
        return pd.DataFrame(rows)


def run_backtest(
    cfg: Config,
    primary_1m: pd.DataFrame,
    correlated_1m: pd.DataFrame | None = None,
    lookahead_minutes: int = 24 * 60,
) -> BacktestResult:
    """Walk every 4H session open in ``primary_1m`` and run the detector,
    then simulate any setup that was taken against the bars that follow.
    """
    session_opens = data_mod.four_hour_bars(primary_1m, cfg.session).index
    daily = data_mod.daily_bars(primary_1m, cfg.session)
    weekly = data_mod.weekly_bars(primary_1m, cfg.session)
    bars_30m = data_mod.minute_bars(primary_1m, 30, cfg.session)

    result = BacktestResult()

    trades_today = 0
    losses_today = 0
    current_date = None

    for session_open in session_opens:
        local_date = session_open.tz_convert(cfg.session.timezone).date()
        if local_date != current_date:
            current_date = local_date
            trades_today = 0
            losses_today = 0

        if (
            trades_today >= cfg.limits.max_trades_per_day
            or losses_today >= cfg.limits.max_losses_per_day
        ):
            continue

        window_end = session_open + pd.Timedelta(minutes=cfg.session.session_cutoff_minutes + 60)
        frames_1m = {"primary": primary_1m[primary_1m.index <= window_end]}
        if correlated_1m is not None:
            frames_1m["correlated"] = correlated_1m[correlated_1m.index <= window_end]
        bias_frames = {"daily": daily, "weekly": weekly, "30min": bars_30m}

        setup = det.detect_session_setup(cfg, session_open, frames_1m, bias_frames)
        result.setups.append(setup)

        if not setup.taken or setup.confirmation_time is None:
            continue

        after = primary_1m[
            (primary_1m.index > setup.confirmation_time)
            & (primary_1m.index <= setup.confirmation_time + pd.Timedelta(minutes=lookahead_minutes))
        ]
        if after.empty:
            continue

        trade = _simulate_trade(setup, after, cfg)
        if trade is not None:
            result.trades.append(trade)
            trades_today += 1
            if not trade.win:
                losses_today += 1

    return result
