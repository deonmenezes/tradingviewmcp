"""Pluggable setup grading.

# TODO: rule unspecified — the source material references a separate A+/A/B/C
# setup ranking rubric that was never provided alongside the mechanical rules.
# This module does NOT invent thresholds for that rubric. score_setup()
# returns a raw weighted score from user-supplied weights (po3.config.
# ScoringWeights) and otherwise reports grade="ungraded". Supply your own
# weights and thresholds before relying on the grade for anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from po3.config import ScoringWeights
from po3.detector import SetupResult

Grade = str  # "ungraded" unless the user defines thresholds elsewhere

_POI_TIMEFRAME_RANK = {"15min": 3, "5min": 2, "1min": 1, "n/a": 0}


@dataclass
class SetupScore:
    raw_score: float
    grade: Grade
    components: dict[str, float]


def score_setup(
    setup: SetupResult,
    weights: ScoringWeights,
    smt_present: bool = False,
    bias_clarity: float = 0.0,
    displacement_strength: float = 0.0,
) -> SetupScore:
    """Weighted sum of the factors the source material says matter for
    grading (POI timeframe quality, SMT confluence, displacement strength,
    bias clarity) using whatever weights the caller supplies.

    Returns grade="ungraded" by default rather than guessing A+/A/B/C
    thresholds — those are not specified in the source rules.
    """
    poi_tf = setup.poi.timeframe if setup.poi is not None else "n/a"
    poi_component = _POI_TIMEFRAME_RANK.get(poi_tf, 0) * weights.poi_timeframe_weight
    smt_component = (1.0 if smt_present else 0.0) * weights.smt_present_weight
    displacement_component = displacement_strength * weights.displacement_strength_weight
    clarity_component = bias_clarity * weights.bias_clarity_weight

    components = {
        "poi_timeframe": poi_component,
        "smt_present": smt_component,
        "displacement_strength": displacement_component,
        "bias_clarity": clarity_component,
    }
    raw_score = sum(components.values())

    return SetupScore(raw_score=raw_score, grade="ungraded", components=components)
