# PO3 / Power-of-Three Setup Detector & Backtester

Mechanical, rule-based detector and backtester for the "Power of Three"
(AMD: Accumulation → Manipulation → Distribution) trading model, built for
NQ (primary) with ES as a correlated asset for SMT divergence checks.

This encodes the rules **as described** in the source course material. It
is not a claim that the strategy is profitable — run it on real data and
judge the measured stats yourself.

## Install

```
pip install -r requirements.txt
```

## Run

```
python -m po3.cli --primary-csv path/to/NQ_1m.csv --correlated-csv path/to/ES_1m.csv --explain
```

Input CSVs: `timestamp, open, high, low, close, volume`, one file per
asset, 1-minute resolution (everything else is derived from this — see
`po3/data.py`). Use `--input-tz` if timestamps are naive and not UTC.

## Tests

```
pytest tests/
```

## Module map

- `config.py` — all tunables (sessions, risk, POI priority, confirmation
  tiers, calendar filters). Several fields are explicitly marked
  `# TODO: rule unspecified` for rubrics the source material references but
  never defines (see `scoring.py` docstring) — supply your own values.
- `data.py` — 1m loading + session-aligned resampling to 5m/15m/1H/4H/D/W.
- `primitives.py` — FVG, IFVG, CISD, liquidity sweep, BPR, SMT divergence.
- `bias.py` — Daily/Weekly/30m PO3 bias + OHLC "genetic makeup" heuristic.
- `detector.py` — assembles one session's setup: time window → bias → POI
  → confirmation tier → stop/targets, recording a gate-by-gate trail.
- `backtest.py` — walks every 4H session open, simulates trade management
  (breakeven move, scaled partials) and R accounting.
- `stats.py` — win rate, expectancy, profit factor, splits by session/
  weekday/POI type.
- `report.py` — trade log CSV, equity curve PNG, P&L calendar CSV,
  `--explain` text.
- `scoring.py` — pluggable `score_setup()`; returns `grade="ungraded"` by
  default since the A+/A/B/C rubric is not specified in the source.

## Known limitations / explicitly excluded

- Top/bottom-ticking (anticipation entries) — discretionary, excluded.
- Continuation entries with no clean manipulation — opt-in, off by default
  (`config.ExperimentalConfig.allow_continuation_entries`).
- `candle_bias()` infers which wick formed first from OHLC alone (close
  position within the bar's range); true intrabar sequencing needs
  tick/lower-TF data and isn't available from OHLC bars.
- `sample_data/generate_sample.py` produces synthetic random-walk data for
  smoke-testing the pipeline only — it is not real market data and proves
  nothing about strategy performance.
