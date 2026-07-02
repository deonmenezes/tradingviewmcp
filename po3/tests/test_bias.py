import pandas as pd

from po3.bias import candle_bias, daily_po3_bias, resolve_bias


def _daily(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1D", tz="UTC")
    return pd.DataFrame(rows, index=idx)


def test_candle_bias_close_in_top_half_is_long():
    candle = pd.Series({"open": 100, "high": 110, "low": 90, "close": 108})
    assert candle_bias(candle) == "long"


def test_candle_bias_close_in_bottom_half_is_short():
    candle = pd.Series({"open": 100, "high": 110, "low": 90, "close": 92})
    assert candle_bias(candle) == "short"


def test_daily_po3_bias_uses_most_recent_completed_day():
    df = _daily(
        [
            {"open": 100, "high": 110, "low": 90, "close": 92},
            {"open": 95, "high": 105, "low": 94, "close": 103},
        ]
    )
    as_of = df.index[-1] + pd.Timedelta(hours=1)
    assert daily_po3_bias(df, as_of) == "long"


def test_resolve_bias_unanimous_gives_full_clarity():
    daily = _daily([{"open": 100, "high": 110, "low": 90, "close": 108}])
    weekly = _daily([{"open": 100, "high": 110, "low": 90, "close": 108}])
    as_of = daily.index[-1] + pd.Timedelta(hours=1)
    result = resolve_bias(daily, as_of, weekly_bars=weekly)
    assert result.bias == "long"
    assert result.clarity == 1.0


def test_resolve_bias_disagreement_is_mixed():
    daily = _daily([{"open": 100, "high": 110, "low": 90, "close": 108}])
    weekly = _daily([{"open": 100, "high": 110, "low": 90, "close": 92}])
    as_of = daily.index[-1] + pd.Timedelta(hours=1)
    result = resolve_bias(daily, as_of, weekly_bars=weekly)
    assert result.bias == "mixed"
