import pandas as pd

from po3.primitives import (
    body_closes_through,
    detect_bpr,
    detect_cisd,
    detect_fvgs,
    detect_ifvg,
    detect_liquidity_sweep,
    mark_mitigations,
    smt_divergence,
    unmitigated_fvgs,
)


def _bars(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, index=idx)


def test_bullish_fvg_detected():
    df = _bars(
        [
            {"open": 100, "high": 102, "low": 99, "close": 101},  # c1
            {"open": 101, "high": 106, "low": 101, "close": 105},  # c2 displaces up
            {"open": 105, "high": 108, "low": 104, "close": 107},  # c3 low(104) > c1 high(102)
        ]
    )
    fvgs = detect_fvgs(df, "1min")
    assert len(fvgs) == 1
    assert fvgs[0].direction == "bullish"
    assert fvgs[0].top == 104
    assert fvgs[0].bottom == 102


def test_bearish_fvg_detected():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 98, "close": 99},  # c1
            {"open": 99, "high": 99, "low": 94, "close": 95},  # c2 displaces down
            {"open": 95, "high": 96, "low": 92, "close": 93},  # c3 high(96) < c1 low(98)
        ]
    )
    fvgs = detect_fvgs(df, "1min")
    assert len(fvgs) == 1
    assert fvgs[0].direction == "bearish"
    assert fvgs[0].top == 98
    assert fvgs[0].bottom == 96


def test_no_fvg_when_no_gap():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100.5},
            {"open": 100.5, "high": 101.5, "low": 100, "close": 101},
            {"open": 101, "high": 101.8, "low": 100.2, "close": 101.5},
        ]
    )
    assert detect_fvgs(df, "1min") == []


def test_mitigation_marks_gap_filled():
    df = _bars(
        [
            {"open": 100, "high": 102, "low": 99, "close": 101},
            {"open": 101, "high": 106, "low": 101, "close": 105},
            {"open": 105, "high": 108, "low": 104, "close": 107},
            {"open": 107, "high": 107, "low": 101, "close": 103},  # trades back through gap
        ]
    )
    fvgs = detect_fvgs(df, "1min")
    mark_mitigations(fvgs, df)
    assert fvgs[0].mitigated is True
    assert unmitigated_fvgs(fvgs) == []


def test_body_closes_through_requires_body_not_wick():
    bullish_wick_only = pd.Series({"open": 99, "high": 101.5, "low": 98, "close": 99.5})
    assert body_closes_through(bullish_wick_only, 100.0, "bullish") is False

    bullish_body_close = pd.Series({"open": 99.5, "high": 102, "low": 99, "close": 101.5})
    assert body_closes_through(bullish_body_close, 100.0, "bullish") is True


def test_ifvg_requires_body_close_through_opposing_fvg():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 98, "close": 99},  # c1 of bearish fvg
            {"open": 99, "high": 99, "low": 94, "close": 95},
            {"open": 95, "high": 96, "low": 92, "close": 93},  # c3, bearish fvg top=98 bottom=96
            {"open": 95, "high": 97, "low": 94, "close": 96.5},  # wick only, no body close
            {"open": 96.5, "high": 99.5, "low": 96, "close": 99},  # body closes above 98? no -> 99>98 true
        ]
    )
    fvgs = detect_fvgs(df.iloc[:3], "1min")
    bearish_fvg = fvgs[0]
    ts = detect_ifvg(df, bearish_fvg, "bullish")
    assert ts == df.index[4]


def test_cisd_long_after_down_run():
    df = _bars(
        [
            {"open": 110, "high": 111, "low": 108, "close": 109},  # down candle, run start, open=110
            {"open": 109, "high": 109.5, "low": 106, "close": 107},  # down candle
            {"open": 107, "high": 107.5, "low": 104, "close": 105},  # down candle
            {"open": 105, "high": 109, "low": 104.5, "close": 106},  # up but doesn't close above 110
            {"open": 106, "high": 111.5, "low": 105.5, "close": 111},  # closes above 110 -> CISD
        ]
    )
    ts = detect_cisd(df, "bearish")
    assert ts == df.index[4]


def test_cisd_returns_none_without_run():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 99, "close": 100.5},
            {"open": 100.5, "high": 101.5, "low": 100, "close": 101},
        ]
    )
    assert detect_cisd(df, "bearish") is None


def test_liquidity_sweep_bullish():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 95, "close": 100.5},  # wick below 96, close above
        ]
    )
    ts = detect_liquidity_sweep(df, 96.0, "bullish")
    assert ts == df.index[0]


def test_liquidity_sweep_none_when_no_close_back():
    df = _bars(
        [
            {"open": 100, "high": 101, "low": 95, "close": 95.5},
        ]
    )
    assert detect_liquidity_sweep(df, 96.0, "bullish") is None


def test_bpr_detects_overlapping_opposing_fvgs():
    bull = _bars(
        [
            {"open": 100, "high": 102, "low": 99, "close": 101},
            {"open": 101, "high": 106, "low": 101, "close": 105},
            {"open": 105, "high": 108, "low": 104, "close": 107},
        ]
    )
    bull_fvgs = detect_fvgs(bull, "1min")

    bear = _bars(
        [
            {"open": 105, "high": 106, "low": 103, "close": 104},
            {"open": 104, "high": 104, "low": 99, "close": 100},
            {"open": 100, "high": 102.5, "low": 98, "close": 99},
        ]
    )
    bear_fvgs = detect_fvgs(bear, "1min")

    pairs = detect_bpr(bull_fvgs, bear_fvgs)
    assert len(pairs) == 1


def test_smt_divergence_bearish_at_high():
    primary = _bars(
        [
            {"open": 100, "high": 105, "low": 99, "close": 102},
            {"open": 102, "high": 108, "low": 101, "close": 107},  # new high
        ]
    )
    correlated = _bars(
        [
            {"open": 50, "high": 55, "low": 49, "close": 52},
            {"open": 52, "high": 54, "low": 51, "close": 53},  # lower high
        ]
    )
    assert smt_divergence(primary, correlated, "high") is True


def test_smt_divergence_false_when_aligned():
    primary = _bars(
        [
            {"open": 100, "high": 105, "low": 99, "close": 102},
            {"open": 102, "high": 108, "low": 101, "close": 107},
        ]
    )
    correlated = _bars(
        [
            {"open": 50, "high": 55, "low": 49, "close": 52},
            {"open": 52, "high": 58, "low": 51, "close": 57},
        ]
    )
    assert smt_divergence(primary, correlated, "high") is False
