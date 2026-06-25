"""Central configuration for the PO3 detector/backtester.

All "magic numbers" referenced in the spec live here so behavior can be
tuned without touching detection logic. Anything marked
``# TODO: rule unspecified`` is a deliberate stub for a rule the source
material references but never fully defines (see spec §9) — the user must
supply real values/weights rather than the tool guessing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BiasMode = Literal["long", "short", "auto"]


@dataclass(frozen=True)
class SessionConfig:
    """Session/timezone boundaries. Every higher-timeframe alignment in this
    strategy is derived from these, never from pre-aggregated bars, because
    misaligned session boundaries are the single biggest source of bugs.
    """

    timezone: str = "America/New_York"
    session_open_hour: int = 18  # 18:00 ET daily session open
    four_hour_opens: tuple[int, ...] = (18, 22, 2, 6, 10, 14)
    tradeable_opens: tuple[int, ...] = (10, 14)
    macro_window_minutes: int = 10  # last N min + first N min of each clock hour
    session_cutoff_minutes: int = 150  # stand down if no entry within N min of open


@dataclass(frozen=True)
class RiskConfig:
    breakeven_r: float = 1.25  # move stop to BE around 1R-1.5R
    trim_fractions: tuple[float, ...] = (0.6, 0.2, 0.2)  # TP1, TP2, TP3 (sums to 1.0)
    trim_r_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)
    tighten_stop_to_order_block: bool = False
    allow_single_candle_cisd: bool = False


@dataclass(frozen=True)
class POIConfig:
    """Priority order and toggles for manipulation target resolution (§4.3)."""

    fvg_timeframes_priority: tuple[str, ...] = ("15min", "5min", "1min")
    reversal_extra_timeframes: tuple[str, ...] = ("1H", "4H", "1D")
    allow_sweep_fallback: bool = True  # used <5% of the time per the source material
    bpr_is_high_quality: bool = True


@dataclass(frozen=True)
class ConfirmationConfig:
    """Entry confirmation tier selection (§4.4)."""

    require_tier3_when_mixed_bias: bool = True
    smt_required: bool = False  # optional confluence, off by default


@dataclass(frozen=True)
class ScoringWeights:
    """User-supplied weights for score_setup(). Defaults are neutral
    placeholders, not a claim about real predictive value.

    # TODO: rule unspecified — the source references a separate A+/A/B/C
    # setup ranking rubric that is not provided. These weights and the
    # resulting grade thresholds must be supplied/tuned by the user.
    """

    poi_timeframe_weight: float = 0.0
    smt_present_weight: float = 0.0
    displacement_strength_weight: float = 0.0
    bias_clarity_weight: float = 0.0


@dataclass(frozen=True)
class CalendarConfig:
    """News/calendar filter toggles (§5). Hooks only — the user supplies the
    actual calendar CSV at runtime.
    """

    skip_bank_holidays: bool = True
    skip_day_before_fomc: bool = True
    skip_fomc_day: bool = True
    skip_day_before_cpi: bool = True
    minutes_before_red_release_blackout: int = 15
    # TODO: rule unspecified — "not instantly high-volatility" on CPI day
    # cannot be fully automated; expose as a manual threshold the user sets.
    cpi_volatility_threshold_pct: float | None = None
    flag_mag7_earnings_day_before: bool = True


@dataclass(frozen=True)
class ExperimentalConfig:
    """Discretionary paths excluded from the mechanical backtest by default."""

    allow_top_bottom_ticking: bool = False  # explicitly discretionary, excluded
    allow_continuation_entries: bool = False  # opt-in, off by default


@dataclass(frozen=True)
class Config:
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    poi: POIConfig = field(default_factory=POIConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)
    bias_mode: BiasMode = "auto"
    smt_asset: str | None = "ES"


DEFAULT_CONFIG = Config()
