"""Bootstrap Monte Carlo resampling over a backtest's realized R-multiples.

This does not simulate new trades or new market conditions — it resamples
(with replacement) the R-multiples a backtest already realized, to show how
much the single observed equity curve could vary by reshuffling trade order
and trade-selection luck. It measures variance in the historical trade
distribution, not a guarantee of future edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    n_simulations: int
    n_trades_per_sim: int
    final_return_pct: np.ndarray  # one final cumulative return % per simulation
    max_drawdown_pct: np.ndarray  # one max drawdown % per simulation

    def summary(self) -> dict:
        prob_positive = float((self.final_return_pct > 0).mean())
        return {
            "n_simulations": self.n_simulations,
            "n_trades_per_sim": self.n_trades_per_sim,
            "prob_positive_return": prob_positive,
            "return_pct_p5": float(np.percentile(self.final_return_pct, 5)),
            "return_pct_p25": float(np.percentile(self.final_return_pct, 25)),
            "return_pct_p50": float(np.percentile(self.final_return_pct, 50)),
            "return_pct_p75": float(np.percentile(self.final_return_pct, 75)),
            "return_pct_p95": float(np.percentile(self.final_return_pct, 95)),
            "max_drawdown_pct_p50": float(np.percentile(self.max_drawdown_pct, 50)),
            "max_drawdown_pct_p95": float(np.percentile(self.max_drawdown_pct, 95)),
        }


def run_monte_carlo(
    trade_log: pd.DataFrame,
    risk_per_trade_pct: float,
    n_simulations: int = 5000,
    n_trades_per_sim: int | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Resample ``trade_log["r_achieved"]`` with replacement ``n_simulations``
    times to build a distribution of equity-curve outcomes.

    Each R-multiple is converted to a percent account return via
    ``risk_per_trade_pct`` (e.g. r=2.0 at 0.5% risk -> +1.0% equity move),
    compounded multiplicatively trade-by-trade within each simulated path.
    """
    r_values = trade_log["r_achieved"].dropna().to_numpy()
    if r_values.size == 0:
        raise ValueError("trade_log has no r_achieved values to resample")

    n_trades = n_trades_per_sim or r_values.size
    rng = np.random.default_rng(seed)

    sampled = rng.choice(r_values, size=(n_simulations, n_trades), replace=True)
    per_trade_return = 1.0 + sampled * (risk_per_trade_pct / 100.0)
    equity_paths = np.cumprod(per_trade_return, axis=1)

    final_return_pct = (equity_paths[:, -1] - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity_paths, axis=1)
    drawdown = (equity_paths - running_max) / running_max
    max_drawdown_pct = drawdown.min(axis=1) * -100.0

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades_per_sim=n_trades,
        final_return_pct=final_return_pct,
        max_drawdown_pct=max_drawdown_pct,
    )
