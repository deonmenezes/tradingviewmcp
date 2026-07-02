"""Generate small synthetic NQ/ES 1-minute OHLCV CSVs for a smoke-test run.

This is NOT real market data — it exists only to exercise the pipeline
end-to-end (resampling, primitives, detector, backtest, report) and to
produce something for `--explain` to narrate. Do not draw any conclusions
about strategy performance from this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate(symbol_seed: int, days: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    periods = days * 24 * 60
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC")

    price = 15000.0 if symbol_seed == 1 else 4500.0
    rows = []
    for ts in idx:
        drift = rng.normal(0, 0.6)
        local_hour = ts.tz_convert("America/New_York").hour
        if local_hour in (10, 14):
            drift += rng.choice([-1, 1]) * rng.uniform(1.5, 4.0)
        price = max(price + drift, 1.0)
        o = price
        h = o + abs(rng.normal(1.0, 1.0))
        low = o - abs(rng.normal(1.0, 1.0))
        c = rng.uniform(low, h)
        vol = abs(rng.normal(500, 100))
        rows.append((ts, o, h, low, c, vol))
        price = c

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df


if __name__ == "__main__":
    nq = generate(symbol_seed=1, days=10)
    es = generate(symbol_seed=2, days=10)
    nq.to_csv("sample_data/NQ_1m.csv", index=False)
    es.to_csv("sample_data/ES_1m.csv", index=False)
    print("wrote sample_data/NQ_1m.csv and sample_data/ES_1m.csv")
