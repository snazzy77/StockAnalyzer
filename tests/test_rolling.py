import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_analyzer.regression import features
from stock_analyzer.rolling import fit_recent, recent_training, run_rolling, walk_forward, render_report


class RollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(6)
        dates = pd.bdate_range("2018-01-01", periods=1600)
        prices = pd.DataFrame({name: 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates)))) for name in ("NVDA", "SPY", "AMD")}, index=dates)
        cls.prices = prices
        cls.volume = pd.Series(rng.integers(1000, 5000, len(dates)), index=dates)
        cls.frame = features(prices, cls.volume, "NVDA", 63)

    def test_enhanced_features_are_causal_and_macro_is_lagged(self):
        prices = self.prices.copy()
        prices["SOXX"] = prices.AMD
        prices["^TNX"] = np.arange(len(prices), dtype=float) + 1
        prices["DX-Y.NYB"] = prices.SPY * 2
        frame = features(prices, self.volume, "NVDA", 5, sector="SOXX")
        i = 900
        self.assertAlmostEqual(frame.enh_yield_change_5.iloc[i], 5)
        self.assertAlmostEqual(frame.enh_sector_relative_5.iloc[i],
                               prices.NVDA.iloc[i] / prices.NVDA.iloc[i-5] - prices.SOXX.iloc[i] / prices.SOXX.iloc[i-5])
        changed = prices.copy()
        changed.loc[changed.index[i]:, ["^TNX", "DX-Y.NYB"]] *= 10
        other = features(changed, self.volume, "NVDA", 5, sector="SOXX")
        self.assertEqual(frame.enh_yield_change_5.iloc[i], other.enh_yield_change_5.iloc[i])
        self.assertEqual(frame.enh_dollar_return_5.iloc[i], other.enh_dollar_return_5.iloc[i])
        changed = prices.copy()
        changed.iloc[i+1:] *= 10
        other = features(changed, self.volume, "NVDA", 5, sector="SOXX")
        cols = [c for c in frame if c.startswith("enh_")]
        pd.testing.assert_frame_equal(frame.loc[:prices.index[i], cols], other.loc[:prices.index[i], cols])
        self.assertNotIn("peer_^TNX", frame)

    def test_long_windows_and_period_diagnostics(self):
        report, coefficients, predictions, _ = run_rolling(self.frame, "2020-12-31", "2022-12-31", 63,
                                               quarters=(4, 8), alphas=(10.0,))
        report["ticker"] = "NVDA"
        rendered = render_report(report, coefficients)
        self.assertIn("Signal diagnostics", rendered)
        self.assertIn("Test 2023", rendered)
        self.assertIn("ridge_8q_a10_enhanced", report["comparisons"])
        for comparison in report["comparisons"].values():
            self.assertEqual(len(comparison["validation_periods"]), 3)
            self.assertIn("rank_ic", comparison["test"])
            self.assertLess(comparison["test_nonoverlap"]["rows"], comparison["test"]["rows"])
            for period in comparison["validation_periods"]:
                start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
                self.assertLessEqual(self.frame.loc[end, "target_end"], pd.Timestamp("2022-12-31"))
                self.assertLessEqual(start, end)
        self.assertEqual(report["evaluation_common_rows"]["test"], int((predictions.evaluation_split == "test").sum()))

    def test_window_does_not_reach_back_for_unfinished_labels(self):
        asof = self.frame.index[1000]
        train = recent_training(self.frame, asof, 2)
        self.assertGreaterEqual(train.index.min(), self.frame.index[875])
        self.assertTrue((train.target_end <= asof).all())
        self.assertEqual(len(train), 63)

    def test_future_prices_cannot_change_historical_fit(self):
        asof = self.frame.index[1000]
        changed = self.prices.copy()
        changed.loc[changed.index > asof] *= 10
        other = features(changed, self.volume, "NVDA", 63)
        first = fit_recent(self.frame, asof, 4, 10, 63)
        second = fit_recent(other, asof, 4, 10, 63)
        np.testing.assert_allclose(first.model.coef_, second.model.coef_)
        self.assertEqual(first.audit, second.audit)

    def test_outcomes_older_than_window_do_not_affect_fit(self):
        asof = self.frame.index[1000]
        changed = self.frame.copy()
        changed.loc[changed.index < self.frame.index[749], "target"] = 100.0
        first = fit_recent(self.frame, asof, 4, 10, 63)
        second = fit_recent(changed, asof, 4, 10, 63)
        np.testing.assert_allclose(first.model.coef_, second.model.coef_)
        self.assertEqual(first.baseline, second.baseline)

    def test_monthly_schedule_and_latest_fit(self):
        dates = self.frame.index[800:870]
        _, audits = walk_forward(self.frame, dates, 4, 10, 63, 2)
        self.assertEqual(len(audits), len(dates.to_period("M").unique()))
        report, _, predictions, bundle = run_rolling(self.frame, "2020-12-31", "2022-12-31", 63,
                                                    quarters=(2, 3, 4), alphas=(10.0,))
        self.assertEqual(bundle.audit["asof"], str(self.frame.index[-1].date()))
        self.assertLessEqual(pd.Timestamp(bundle.audit["last_target_end"]), self.frame.index[-1])
        self.assertLessEqual(len(bundle.mean), 6 + len([c for c in self.frame if c.startswith("enh_")]))
        self.assertFalse(report["selected_variant"].startswith("benchmark"))
        self.assertAlmostEqual(sum(report["latest_contributions"].values()), report["latest_expected_return"])
        self.assertIn("test", predictions.evaluation_split.values)

    def test_test_outcomes_do_not_select_configuration(self):
        first = run_rolling(self.frame, "2020-12-31", "2022-12-31", 63, quarters=(4,), alphas=(10.0,))[0]
        modified = self.frame.copy()
        modified.loc[modified.target_end > "2022-12-31", "target"] += 3
        second = run_rolling(modified, "2020-12-31", "2022-12-31", 63, quarters=(4,), alphas=(10.0,))[0]
        self.assertEqual(first["selected_variant"], second["selected_variant"])
        for key in first["comparisons"]:
            self.assertEqual(first["comparisons"][key]["validation"], second["comparisons"][key]["validation"])

    def test_baseline_fallback_wins_when_ridge_adds_nothing(self):
        frame = self.frame.copy()
        frame.loc[frame.target.notna(), "target"] = 0.0
        report, coefficients, _, bundle = run_rolling(frame, "2020-12-31", "2022-12-31", 63,
                                                     quarters=(4,), alphas=(10.0,), max_peers=0)
        self.assertEqual(report["selected_variant"], "mean_4q")
        self.assertEqual(report["signal_status"], "no_useful_signal")
        self.assertIsNone(bundle.model)
        self.assertEqual(len(coefficients), 1)


if __name__ == "__main__":
    unittest.main()
