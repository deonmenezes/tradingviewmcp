"""Assemble a PO3 setup at a single session open from the primitives.

This is the deterministic gate chain described in spec §4: time window ->
bias -> manipulation/POI -> confirmation tier -> stop/targets. Every gate
records why it passed or failed so `--explain` can print a readable trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from po3 import bias as bias_mod
from po3 import data as data_mod
from po3 import primitives as prim
from po3.config import Config

Direction = Literal["long", "short"]
Tier = Literal["tier1_ifvg", "tier2_cisd", "tier3_reversal", "sweep_fallback"]


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class SetupResult:
    session_open: pd.Timestamp
    direction: Direction | None
    taken: bool
    gates: list[GateResult] = field(default_factory=list)
    poi: prim.FVG | None = None
    confirmation_tier: Tier | None = None
    confirmation_time: pd.Timestamp | None = None
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = field(default_factory=list)
    manipulation_extreme: float | None = None

    def explain(self) -> str:
        lines = [f"Session open: {self.session_open} | taken={self.taken}"]
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{mark}] {g.name}: {g.detail}")
        if self.taken:
            lines.append(
                f"  -> {self.direction} entry={self.entry} stop={self.stop} "
                f"targets={self.targets} tier={self.confirmation_tier}"
            )
        return "\n".join(lines)


def _is_tradeable_open(ts: pd.Timestamp, cfg: Config) -> bool:
    local_hour = ts.tz_convert(cfg.session.timezone).hour
    return local_hour in cfg.session.tradeable_opens


def _opposing_direction(d: Direction) -> Direction:
    return "short" if d == "long" else "long"


def _fvg_direction_for_entry(entry_direction: Direction) -> prim.Direction:
    """The opposing FVG type we look to invert/sweep for this entry
    direction: longs invert/sweep bearish FVGs, shorts invert bullish FVGs.
    """
    return "bearish" if entry_direction == "long" else "bullish"


def resolve_poi(
    cfg: Config,
    price: float,
    entry_direction: Direction,
    frames: dict[str, pd.DataFrame],
    accumulation_high: float,
    accumulation_low: float,
) -> tuple[prim.FVG | None, str]:
    """Manipulation target priority chain (§4.3). Returns the chosen POI
    (or None) and a human-readable reason for the choice.
    """
    opposing = _fvg_direction_for_entry(entry_direction)

    for tf in cfg.poi.fvg_timeframes_priority:
        df = frames.get(tf)
        if df is None or df.empty:
            continue
        fvgs = prim.detect_fvgs(df, tf)
        prim.mark_mitigations(fvgs, df)
        poi = prim.closest_unmitigated_fvg(fvgs, price, opposing)
        if poi is not None:
            return poi, f"unmitigated {opposing} FVG on {tf} closest to price"

    if cfg.poi.allow_sweep_fallback:
        return None, "no unmitigated FVG on any timeframe; fall back to liquidity sweep"

    return None, "no unmitigated FVG and sweep fallback disabled"


def confirm_entry(
    cfg: Config,
    df_entry_tf: pd.DataFrame,
    entry_direction: Direction,
    poi: prim.FVG | None,
    bias_clarity: float,
) -> tuple[Tier | None, pd.Timestamp | None]:
    """Pick and evaluate the confirmation tier per §4.4. Returns the tier
    used and the timestamp confirmation occurred, or (None, None).
    """
    manipulation_direction = _fvg_direction_for_entry(entry_direction)
    entry_dir_prim: prim.Direction = "bullish" if entry_direction == "long" else "bearish"

    need_tier3 = cfg.confirmation.require_tier3_when_mixed_bias and bias_clarity < 1.0

    if poi is not None:
        ifvg_ts = prim.detect_ifvg(df_entry_tf, poi, entry_dir_prim)
        if ifvg_ts is not None and not need_tier3:
            return "tier1_ifvg", ifvg_ts

    cisd_ts = prim.detect_cisd(
        df_entry_tf,
        manipulation_direction,
        allow_single_candle=cfg.risk.allow_single_candle_cisd,
    )
    if cisd_ts is not None and not need_tier3:
        return "tier2_cisd", cisd_ts

    if need_tier3 and (poi is not None) and cisd_ts is not None:
        after_first_confirm = df_entry_tf[df_entry_tf.index > cisd_ts]
        new_fvgs = prim.detect_fvgs(after_first_confirm, "tier3")
        prim.mark_mitigations(new_fvgs, after_first_confirm)
        same_dir_fvg = prim.closest_unmitigated_fvg(
            new_fvgs, df_entry_tf["close"].iloc[-1], entry_dir_prim
        )
        if same_dir_fvg is not None:
            second_cisd = prim.detect_cisd(
                after_first_confirm[after_first_confirm.index > same_dir_fvg.end_time],
                manipulation_direction,
                allow_single_candle=cfg.risk.allow_single_candle_cisd,
            )
            if second_cisd is not None:
                return "tier3_reversal", second_cisd

    return None, None


def compute_targets(
    entry: float,
    stop: float,
    direction: Direction,
    daily_fvgs: list[prim.FVG],
    fallback_fvgs: list[list[prim.FVG]],
) -> list[float]:
    """Target priority chain (§4.5): nearest Daily FVG in trade direction,
    then 4H/1H FVGs, then a Fibonacci-extension stretch target.
    """
    same_dir: prim.Direction = "bullish" if direction == "long" else "bearish"
    targets: list[float] = []

    for fvg in sorted(
        prim.unmitigated_fvgs(daily_fvgs, same_dir),
        key=lambda f: abs(f.midpoint - entry),
    ):
        level = fvg.bottom if direction == "long" else fvg.top
        if (direction == "long" and level > entry) or (direction == "short" and level < entry):
            targets.append(level)
            break

    if not targets:
        for fvgs in fallback_fvgs:
            candidates = sorted(
                prim.unmitigated_fvgs(fvgs, same_dir), key=lambda f: abs(f.midpoint - entry)
            )
            for fvg in candidates:
                level = fvg.bottom if direction == "long" else fvg.top
                if (direction == "long" and level > entry) or (
                    direction == "short" and level < entry
                ):
                    targets.append(level)
                    break
            if targets:
                break

    risk = abs(entry - stop)
    stretch = entry + risk * 2.0 if direction == "long" else entry - risk * 2.0
    targets.append(stretch)
    return targets


def detect_session_setup(
    cfg: Config,
    session_open: pd.Timestamp,
    frames_1m: dict[str, pd.DataFrame],
    bias_frames: dict[str, pd.DataFrame],
) -> SetupResult:
    """Evaluate one 4H session open and return whether/how a setup formed.

    frames_1m must contain at least key "primary" (e.g. NQ 1m bars covering
    the session window and lookback). bias_frames should contain "daily",
    optionally "weekly" and "30min", covering history up to session_open.
    """
    result = SetupResult(session_open=session_open, direction=None, taken=False)

    tradeable = _is_tradeable_open(session_open, cfg)
    result.gates.append(
        GateResult("tradeable_open", tradeable, f"hour={session_open.tz_convert(cfg.session.timezone).hour}")
    )
    if not tradeable:
        return result

    primary = frames_1m["primary"]
    cutoff = session_open + pd.Timedelta(minutes=cfg.session.session_cutoff_minutes)
    window = primary[(primary.index >= session_open - pd.Timedelta(minutes=10)) & (primary.index <= cutoff)]
    if window.empty:
        result.gates.append(GateResult("has_data", False, "no bars in session window"))
        return result
    result.gates.append(GateResult("has_data", True, f"{len(window)} 1m bars in window"))

    br = bias_mod.resolve_bias(
        bias_frames["daily"],
        session_open,
        bias_frames.get("weekly"),
        bias_frames.get("30min"),
    )
    if cfg.bias_mode == "auto":
        direction: Direction | None = None if br.bias == "mixed" else br.bias  # type: ignore[assignment]
    else:
        direction = cfg.bias_mode  # type: ignore[assignment]

    result.gates.append(
        GateResult(
            "bias",
            direction is not None,
            f"components={br.components} resolved={br.bias} clarity={br.clarity:.2f}",
        )
    )
    if direction is None:
        return result
    result.direction = direction

    accumulation = window[window.index < session_open]
    if accumulation.empty:
        accumulation_high = float(window["high"].iloc[0])
        accumulation_low = float(window["low"].iloc[0])
    else:
        accumulation_high = float(accumulation["high"].max())
        accumulation_low = float(accumulation["low"].min())

    frames_by_tf = {
        "15min": data_mod.minute_bars(window, 15, cfg.session),
        "5min": data_mod.minute_bars(window, 5, cfg.session),
        "1min": window,
    }
    price_at_open = float(window[window.index >= session_open]["open"].iloc[0]) if (
        window.index >= session_open
    ).any() else float(window["close"].iloc[-1])

    poi, poi_reason = resolve_poi(
        cfg, price_at_open, direction, frames_by_tf, accumulation_high, accumulation_low
    )
    result.poi = poi
    result.gates.append(GateResult("poi_resolved", poi is not None, poi_reason))

    manipulation_extreme = accumulation_low if direction == "long" else accumulation_high
    sweep_level = accumulation_low if direction == "long" else accumulation_high
    sweep_dir: prim.Direction = "bullish" if direction == "long" else "bearish"
    post_open = window[window.index >= session_open]
    sweep_ts = prim.detect_liquidity_sweep(post_open, sweep_level, sweep_dir)

    if poi is None:
        if not cfg.poi.allow_sweep_fallback or sweep_ts is None:
            result.gates.append(GateResult("manipulation", False, "no POI and no liquidity sweep"))
            return result
        result.gates.append(GateResult("manipulation", True, f"sweep fallback at {sweep_ts}"))
        manipulation_extreme = float(
            post_open.loc[sweep_ts, "low" if direction == "long" else "high"]
        )
    else:
        result.gates.append(GateResult("manipulation", True, f"POI tap expected near {poi.midpoint:.2f}"))
        if sweep_ts is not None:
            manipulation_extreme = float(
                post_open.loc[sweep_ts, "low" if direction == "long" else "high"]
            )
    result.manipulation_extreme = manipulation_extreme

    tier, confirm_ts = confirm_entry(cfg, post_open, direction, poi, br.clarity)
    result.gates.append(
        GateResult("confirmation", tier is not None, f"tier={tier} at {confirm_ts}")
    )
    if tier is None:
        return result

    result.confirmation_tier = tier
    result.confirmation_time = confirm_ts
    entry_price = float(post_open.loc[confirm_ts, "close"])
    stop = manipulation_extreme
    result.entry = entry_price
    result.stop = stop

    daily_fvgs = prim.detect_fvgs(bias_frames["daily"], "1D")
    prim.mark_mitigations(daily_fvgs, bias_frames["daily"])
    four_h = data_mod.four_hour_bars(primary, cfg.session)
    one_h = data_mod.minute_bars(primary, 60, cfg.session)
    four_h_fvgs = prim.detect_fvgs(four_h, "4H")
    prim.mark_mitigations(four_h_fvgs, four_h)
    one_h_fvgs = prim.detect_fvgs(one_h, "1H")
    prim.mark_mitigations(one_h_fvgs, one_h)

    targets = compute_targets(entry_price, stop, direction, daily_fvgs, [four_h_fvgs, one_h_fvgs])
    result.targets = targets
    result.taken = True
    return result
