"""Command-line entrypoint: run the PO3 detector/backtester on CSV data."""

from __future__ import annotations

import argparse
import sys

from po3 import data as data_mod
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

    print(
        "\nCaveats: small/limited sample size and potential lookahead bias in this "
        "backtest simulation should be assumed until independently verified. These "
        "are measured stats from the rules as implemented, not a claim that the "
        "strategy is profitable."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
