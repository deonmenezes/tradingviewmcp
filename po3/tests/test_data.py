import pandas as pd

from po3.config import SessionConfig
from po3.data import four_hour_bars


def _make_1m(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10,
        },
        index=idx,
    )


def test_four_hour_bars_open_at_configured_hours():
    session = SessionConfig()
    # 3 full days of 1m bars in UTC starting at midnight.
    df = _make_1m("2024-01-01 00:00", periods=60 * 24 * 3)
    bars = four_hour_bars(df, session)

    local_hours = set(bars.index.tz_convert(session.timezone).hour)
    assert local_hours.issubset(set(session.four_hour_opens))
    assert len(local_hours) > 0
