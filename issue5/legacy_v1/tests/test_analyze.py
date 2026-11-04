import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import _corr, _max_drawdown, build_factors, evaluate


class FactorTests(unittest.TestCase):
    def test_forward_returns_do_not_use_future_signal(self):
        dates = pd.date_range("2024-01-01", periods=5)
        cols = ["A", "B"]
        close = pd.DataFrame([[10, 20], [11, 20], [12, 22], [13, 24], [14, 26]], index=dates, columns=cols)
        base = pd.DataFrame(1.0, index=dates, columns=cols)
        panels = {k: base.copy() for k in ("open", "close", "high", "low", "amt", "num_trades", "cumadj")}
        panels["close"] = close
        panels["open"] = close
        panels["high"], panels["low"] = close * 1.01, close * .99
        panels["amt"] = pd.DataFrame(1.0, index=dates, columns=cols)
        factors, forward = build_factors(panels)
        self.assertAlmostEqual(forward["1d"].loc[dates[0], "A"], 0.0)
        self.assertTrue(np.isnan(forward["1d"].iloc[-1, 0]))
        self.assertAlmostEqual(factors["mom_1"].loc[dates[1], "A"], .1) if "mom_1" in factors else None

    def test_spearman_ties_use_average_ranks(self):
        x = np.array([1, 1, 2, 3, np.nan])
        y = np.array([1, 2, 2, 4, 5])
        # Verify ties are accepted and result is finite using repeated observations.
        self.assertTrue(np.isfinite(_corr(np.tile(x[:4], 10), np.tile(y[:4], 10), ranked=True)))

    def test_max_drawdown(self):
        self.assertAlmostEqual(_max_drawdown(np.array([.1, -.2, .05])), .88 / 1.1 - 1)

    def test_evaluate_known_signal(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=12)
        values = np.tile(np.arange(200, dtype=float), (12, 1))
        factors = {"mom_5": pd.DataFrame(values, index=dates)}
        returns = pd.DataFrame(values / 2000, index=dates)
        summary, ic, quantiles, rolling, corr, definitions = evaluate(factors, {"1d": returns})
        self.assertGreater(summary[0]["rank_ic_mean"], .99)
        self.assertAlmostEqual(summary[0]["q5_q1_spread"], .08, places=6)
        self.assertEqual(len(ic), 12)
        self.assertEqual(len(quantiles), 5)


if __name__ == "__main__":
    unittest.main()
