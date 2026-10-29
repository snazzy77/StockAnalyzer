import sys
import unittest
import importlib.util
import json
import tempfile
import warnings
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_analyzer.regression import features, purged_split, run_regression, select_features
from stock_analyzer.relationships import extract_mentions, sec_candidates


class RegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.dates = pd.bdate_range("2014-01-01", periods=2400)
        self.prices = pd.DataFrame({name: 100 * np.exp(np.cumsum(rng.normal(.0003, .01, len(self.dates))))
                                    for name in ("NVDA", "SPY", "AMD", "XLE")}, index=self.dates)
        self.volume = pd.Series(rng.integers(1000, 5000, len(self.dates)), index=self.dates)

    def test_target_and_latest_unlabeled_rows(self):
        frame = features(self.prices, self.volume, "NVDA", 63)
        self.assertAlmostEqual(frame.target.iloc[300], self.prices.NVDA.iloc[363] / self.prices.NVDA.iloc[300] - 1)
        self.assertTrue(frame.target.tail(63).isna().all())
        self.assertTrue(frame.own_return_21.tail(63).notna().all())

    def test_large_peer_universe_has_no_fragmentation_warning(self):
        peers = pd.DataFrame({f"PEER{i}": self.prices.AMD for i in range(150)})
        prices = pd.concat([self.prices, peers], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            frame = features(prices, self.volume, "NVDA", 63)
        pd.testing.assert_series_equal(frame.peer_PEER149,
            (prices.PEER149.pct_change(21, fill_method=None) - prices.SPY.pct_change(21, fill_method=None)).rename("peer_PEER149"))

    def test_boundaries_purge_overlapping_labels(self):
        frame = features(self.prices, self.volume, "NVDA", 63)
        train, valid, test = purged_split(frame, "2018-12-31", "2020-12-31")
        self.assertTrue((train.target_end <= pd.Timestamp("2018-12-31")).all())
        self.assertTrue((valid.target_end <= pd.Timestamp("2020-12-31")).all())
        self.assertTrue((valid.index > pd.Timestamp("2018-12-31")).all())
        self.assertTrue((test.index > pd.Timestamp("2020-12-31")).all())

    def test_future_changes_do_not_change_features_or_selection(self):
        frame = features(self.prices, self.volume, "NVDA", 21)
        modified = self.prices.copy()
        modified.loc[modified.index > "2018-12-31"] *= 5
        other = features(modified, self.volume, "NVDA", 21)
        train = purged_split(frame, "2018-12-31", "2020-12-31")[0]
        other_train = purged_split(other, "2018-12-31", "2020-12-31")[0]
        pd.testing.assert_frame_equal(train, other_train)
        self.assertEqual(select_features(train)[0], select_features(other_train)[0])

    def test_end_to_end_hac_and_contributions(self):
        for horizon in (21, 63):
            frame = features(self.prices, self.volume, "NVDA", horizon)
            report, coefficients, predictions, bundle = run_regression(frame, "2018-12-31", "2020-12-31", horizon)
            self.assertEqual(bundle[0].cov_type, "HAC")
            self.assertEqual(report["hac_lags"], horizon - 1)
            self.assertAlmostEqual(sum(report["latest_contributions"].values()), report["latest_expected_return"])
            self.assertTrue(np.isfinite(coefficients.coefficient).all())
            self.assertGreater(len(predictions), 20)
            chosen = report["comparisons"][report["selected_variant"]]["test"]
            self.assertEqual(report["beats_test_mean_baseline"], chosen["rmse"] < chosen["baseline_rmse"])

    def test_missing_latest_input_not_replaced_with_older_prediction(self):
        frame = features(self.prices, self.volume, "NVDA", 21)
        frame.loc[frame.index[-1], "own_return_5"] = np.nan
        report, _, _, _ = run_regression(frame, "2018-12-31", "2020-12-31", 21)
        self.assertIsNone(report["latest_expected_return"])

    def test_sec_evidence_is_unverified_and_dated(self):
        companies = [{"ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
                     {"ticker": "META", "title": "META PLATFORMS INC"}]
        html = "<p>Our competitors include Advanced Micro Devices.</p><p>Meta Platforms attended an event.</p>"
        rows = extract_mentions(html, companies, "NVDA", "2020-02-01", "https://www.sec.gov/example")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "AMD")
        self.assertEqual(rows[0]["status"], "unverified_candidate")
        self.assertEqual(rows[0]["known_since"], "2020-02-01")
        self.assertIn("Advanced Micro Devices", rows[0]["evidence"])

    def test_sec_discovery_follows_history_and_excludes_future_filings(self):
        class Client:
            def json(self, url):
                if url.endswith("company_tickers.json"):
                    return {"0": {"ticker": "NVDA", "title": "NVIDIA CORP", "cik_str": 1},
                            "1": {"ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC", "cik_str": 2}}
                if url.endswith("old.json"):
                    return {"filingDate": ["2019-01-01"], "form": ["10-K"],
                            "accessionNumber": ["01-19"], "primaryDocument": ["old.htm"]}
                return {"filings": {"recent": {"filingDate": ["2025-01-01"], "form": ["10-K"],
                       "accessionNumber": ["01-25"], "primaryDocument": ["future.htm"]},
                       "files": [{"filingFrom": "2019-01-01", "name": "old.json"}]}}

            def get(self, url):
                if "future.htm" in url:
                    raise AssertionError("Future filing requested")
                return "<p>Our competitor is Advanced Micro Devices.</p>"

        rows = sec_candidates("NVDA", "2020-01-01", Client())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filing_date"], "2019-01-01")

    def test_offline_cli_writes_reports_and_models(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "regression.py"
        spec = importlib.util.spec_from_file_location("regression_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out" / "nvda"
            out.mkdir(parents=True)
            (out / "candidates.json").write_text(json.dumps({"train_end": "2018-12-31",
                "candidates": [{"ticker": "AMD"}], "warnings": []}))
            cache = root / "cache"
            cache.mkdir()
            for name in ("NVDA", "SPY", "AMD"):
                pd.DataFrame({"Close": self.prices[name], "Volume": self.volume}).to_csv(
                    cache / f"{name}_2014-01-01_2024-01-01.csv")
            argv = ["regression.py", "--mode", "static", "--offline", "--start-date", "2014-01-01", "--end-date", "2024-01-01",
                    "--train-end", "2018-12-31", "--valid-end", "2020-12-31",
                    "--cache-dir", str(cache), "--out-dir", str(root / "out")]
            with patch.object(sys, "argv", argv), patch.object(cli, "discover", side_effect=AssertionError("Network discovery in offline mode")):
                cli.main()
            for horizon in (21, 63):
                report = json.loads((out / "static" / f"{horizon}d.json").read_text())
                self.assertEqual(report["ticker"], "NVDA")
                self.assertTrue((out / "static" / f"{horizon}d_model.joblib").exists())
                self.assertIn("HAC", (out / "static" / f"{horizon}d.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
