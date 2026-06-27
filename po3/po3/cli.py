"""Command-line entrypoint: run the PO3 detector/backtester on CSV data."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from po3 import data as data_mod
from po3 import monte_carlo as mc_mod
from po3 import report
from po3 import stats as stats_mod
from po3.backtest import run_backtest
from po3.config import DEFAULT_CONFIG, Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PO3 / Power-of-Three setup detector & backtester")
    p.add_argument("--primary-csv", required=True, help="Primary asset 1m OHLCV CSV (e.g. NQ)")
    p.add_argument("--correlated-csv", default=None, help="Correlated asset 1m OHLCV CSV (e.g. ES) for SMT")
    p.add_argument("--input-tz", default="UTC", help="Timezone of naive timestamps in the input CSV")
    p.add_argument("--bias", choices=["long", "short", "auto"], default="auto")
    p.add_argument("--out-dir", default="po3_output")
    p.add_argument("--explain", action="store_true", help="Print explain trail for sessions")
    p.add_argument("--explain-limit", type=int, default=10)
    p.add_argument(
        "--start-date",
        default=None,
        help="ISO date (e.g. 2024-06-01); defaults to 12 months before the data's last bar",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="ISO date (e.g. 2025-06-01); defaults to the data's last bar",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Mark this run as pipeline validation only (synthetic/sample data) "
        "and label all printed output accordingly",
    )
    p.add_argument("--monte-carlo", action="store_true", help="Run a bootstrap Monte Carlo over the trade log")
    p.add_argument("--mc-simulations", type=int, default=5000)
    p.add_argument("--mc-seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config(bias_mode=args.bias) if args.bias != DEFAULT_CONFIG.bias_mode else DEFAULT_CONFIG

    primary_1m = data_mod.load_ohlcv_csv(args.primary_csv, tz=args.input_tz)
    correlated_1m = (
        data_mod.load_ohlcv_csv(args.correlated_csv, tz=args.input_tz)
        if args.correlated_csv
        else None
    )

    data_mod.detect_abnormal_jumps(primary_1m, label="primary")
    if correlated_1m is not None:
        data_mod.detect_abnormal_jumps(correlated_1m, label="correlated")
        data_mod.validate_alignment(primary_1m, correlated_1m)

    end_date = pd.Timestamp(args.end_date, tz="UTC") if args.end_date else primary_1m.index[-1]
    start_date = (
        pd.Timestamp(args.start_date, tz="UTC") if args.start_date else end_date - pd.DateOffset(months=12)
    )
    primary_1m = primary_1m[(primary_1m.index >= start_date) & (primary_1m.index <= end_date)]
    if correlated_1m is not None:
        correlated_1m = correlated_1m[(correlated_1m.index >= start_date) & (correlated_1m.index <= end_date)]

    if args.synthetic:
        print(
            "=== SYNTHETIC DATA RUN — pipeline validation only ===\n"
            "Results below come from randomly generated sample data and prove "
            "nothing about real strategy performance. They only confirm the "
            "loader -> detector -> backtest -> stats pipeline runs end-to-end.\n"
        )

    result = run_backtest(cfg, primary_1m, correlated_1m)

    trade_df = report.write_trade_log(result, f"{args.out_dir}/trade_log.csv")
    report.equity_curve(trade_df, f"{args.out_dir}/equity_curve.png")
    report.pnl_calendar(trade_df, f"{args.out_dir}/pnl_calendar.csv")

    overall = stats_mod.overall_stats(trade_df)
    print("=== Overall stats ===")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\n=== By session hour ===")
    print(stats_mod.split_by_session(trade_df))

    print("\n=== By weekday ===")
    print(stats_mod.split_by_weekday(trade_df))

    print("\n=== By POI timeframe ===")
    print(stats_mod.split_by_poi_type(trade_df))

    print(f"\nSessions evaluated: {len(result.setups)} | Setups taken: {len(result.trades)}")

    if args.explain:
        print("\n=== Explain (first sessions) ===")
        print(report.explain_sessions(result.setups, args.explain_limit))

    if args.monte_carlo:
        print("\n=== Monte Carlo (bootstrap resample of realized R-multiples) ===")
        if trade_df.empty:
            print("No trades to resample.")
        else:
            mc_result = mc_mod.run_monte_carlo(
                trade_df,
                risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
                n_simulations=args.mc_simulations,
                seed=args.mc_seed,
            )
            for k, v in mc_result.summary().items():
                print(f"{k}: {v}")
            print(
                "\nMonte Carlo caveat: this resamples the trade R-multiples this "
                "backtest already produced, in random order/combinations. It shows "
                "variance in the historical sample, not a prediction of future "
                "performance, and inherits any bias in the underlying backtest."
            )

    print(
        "\nCaveats: small/limited sample size and potential lookahead bias in this "
        "backtest simulation should be assumed until independently verified. These "
        "are measured stats from the rules as implemented, not a claim that the "
        "strategy is profitable."
    )
    if args.synthetic:
        print(
            "\nReminder: the above was run on SYNTHETIC sample data. It validates "
            "that the pipeline executes correctly — it is not a measurement of "
            "real-world edge."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
