import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_analyzer.reactions import analyze, event_responses, conditional_forecast, render


class ReactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(32)
        dates = pd.bdate_range("2023-01-01", periods=500)
        peer = rng.choice([-.04, .04, 0.0], len(dates))
        own = np.roll(peer, 1) * .7 + rng.normal(0, .001, len(dates))
        own[0] = 0
        cls.prices = pd.DataFrame({"NVDA": 100 * np.cumprod(1 + own), "AMD": 100 * np.cumprod(1 + peer),
                                   "SPY": 100 * np.cumprod(1 + rng.normal(0, .005, len(dates)))}, index=dates)

    def test_future_response_excludes_event_day(self):
        events = event_responses(self.prices, "NVDA", ["AMD"], horizons=(1,))
        jump = events[events.direction == "jump"].iloc[0]
        dive = events[events.direction == "dive"].iloc[0]
        self.assertGreater(jump.mean_return, .025)
        self.assertLess(dive.mean_return, -.025)
        self.assertAlmostEqual(jump.up_frequency, 1.0)

    def test_event_dates_do_not_overlap_and_are_completed(self):
        events = event_responses(self.prices, "NVDA", ["AMD"], horizons=(5,))
        dates = sorted(pd.Timestamp(d) for row in events.itertuples() for d in row.event_dates.split(";") if d)
        positions = [self.prices.index.get_loc(d) for d in dates]
        self.assertTrue(all(b - a >= 5 for a, b in zip(positions, positions[1:])))
        self.assertLessEqual(positions[-1] + 5, len(self.prices) - 1)

    def test_predictive_peer_is_detected_without_causal_claim(self):
        report, comparison, _, model = conditional_forecast(self.prices, "NVDA", ["AMD"], horizon=1)
        self.assertEqual(report["selected_configuration"], "peer_AMD")
        self.assertTrue(report["selected_peer_improved_test_error"])
        self.assertAlmostEqual(sum(report["latest_contributions"].values()), report["latest_expected_return"])
        self.assertLessEqual(pd.Timestamp(model["last_target_end"]), self.prices.index[-1])
        self.assertGreater(next(f for f in report["factors"] if f["factor"] == "peer_AMD")["forecast_change_for_plus_1pp"], 0)

    def test_test_prices_do_not_change_validation_selection(self):
        first = conditional_forecast(self.prices, "NVDA", ["AMD"], horizon=1)
        second_prices = self.prices.copy()
        cutoff = pd.Timestamp(first[0]["evaluation"]["validation_cutoff"])
        second_prices.loc[second_prices.index > cutoff, "NVDA"] *= 1.3
        second = conditional_forecast(second_prices, "NVDA", ["AMD"], horizon=1)
        self.assertEqual(first[0]["selected_configuration"], second[0]["selected_configuration"])
        np.testing.assert_allclose(first[1].validation_rmse, second[1].validation_rmse)

    def test_sec_only_mentions_are_not_assumed_to_be_peers(self):
        prices = self.prices.assign(UNVERIFIED=self.prices.AMD)
        report = analyze(prices, "NVDA", [{"ticker": "AMD", "source": "manual"},
            {"ticker": "UNVERIFIED", "source": "sec_mention"}], forecast_horizon=1)[0]
        self.assertEqual(report["peers"], ["AMD"])
        self.assertEqual(report["excluded_sec_only"], ["UNVERIFIED"])

    def test_report_has_reaction_horizons_baselines_and_scenarios(self):
        result = analyze(self.prices, "NVDA", [{"ticker": "AMD", "source": "manual"}], forecast_horizon=1)
        report = render(*result[:4])
        for horizon in (1, 5, 21):
            self.assertIn(f"| jump | {horizon} |", report)
        self.assertIn("| market_and_own |", report)
        self.assertIn("| mean |", report)
        self.assertIn("forecast_change_for_plus_1pp", str(result[0]))


if __name__ == "__main__":
    unittest.main()
