from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stock_analyzer.backtest import run_threshold_strategy
from stock_analyzer.config import Defaults, Paths
from stock_analyzer.data import fetch_price_history, load_news_csv, save_dataframe
from stock_analyzer.features import FeatureConfig, build_model_dataset
from stock_analyzer.modeling import evaluate, evaluate_with_threshold, train_baseline, tune_threshold
from stock_analyzer.reporting import infer_feature_importance, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline for multiple tickers.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Example: NVDA AAPL MSFT TSLA")
    parser.add_argument("--news-csv", default=str(Paths.raw_data / "news.csv"))
    parser.add_argument("--start-date", default=Defaults.start_date)
    parser.add_argument("--end-date", default=Defaults.end_date)
    parser.add_argument("--horizon-days", type=int, default=Defaults.horizon_days)
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--backtest-start", default="2025-01-01")
    parser.add_argument("--prefer-finbert", action="store_true")
    parser.add_argument("--trade-cost-bps", type=float, default=5.0)
    parser.add_argument("--summary-out", default=str(Paths.reports / "universe_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    news_df = load_news_csv(Path(args.news_csv))
    summary_rows: list[dict] = []

    for ticker in [t.upper() for t in args.tickers]:
        prices = fetch_price_history(ticker, args.start_date, args.end_date)
        ticker_news = news_df.copy()
        if "ticker" in ticker_news.columns:
            ticker_news = ticker_news[
                ticker_news["ticker"].fillna("").str.upper().eq(ticker) | ticker_news["ticker"].isna()
            ]

        dataset = build_model_dataset(
            prices, ticker_news, config=FeatureConfig(horizon_days=args.horizon_days, prefer_finbert=args.prefer_finbert)
        )
        dataset["ticker"] = ticker
        ds_path = Paths.processed_data / f"{ticker.lower()}_dataset.csv"
        save_dataframe(dataset, ds_path)

        train_df = dataset[pd.to_datetime(dataset["date"]) <= pd.to_datetime(args.train_end)].copy()
        valid_df = dataset[
            (pd.to_datetime(dataset["date"]) > pd.to_datetime(args.train_end))
            & (pd.to_datetime(dataset["date"]) <= pd.to_datetime(args.valid_end))
        ].copy()
        test_df = dataset[pd.to_datetime(dataset["date"]) > pd.to_datetime(args.valid_end)].copy()
        if train_df.empty or valid_df.empty or test_df.empty:
            print(f"Skipping {ticker}: insufficient rows across train/valid/test split.")
            continue

        model = train_baseline(train_df)
        valid_metrics, _ = evaluate(model, valid_df)
        test_metrics, _ = evaluate(model, test_df)
        best_threshold, _ = tune_threshold(model, valid_df)
        test_thr = evaluate_with_threshold(model, test_df, best_threshold)

        bt_df = dataset[pd.to_datetime(dataset["date"]) >= pd.to_datetime(args.backtest_start)].copy()
        bt = run_threshold_strategy(bt_df, model, prob_threshold=best_threshold, trade_cost_bps=args.trade_cost_bps)

        model_path = Paths.models / f"{ticker.lower()}_model.joblib"
        import joblib

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        model_metrics = {
            "valid": valid_metrics,
            "test": test_metrics,
            "best_threshold": best_threshold,
            "test_at_best_threshold": test_thr,
            "model_quality_flag": "predictive" if test_metrics.get("roc_auc", 0.0) >= 0.55 else "non_predictive",
        }
        model_metrics_path = Paths.metrics / f"{ticker.lower()}_metrics.json"
        model_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_metrics_path.write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")

        eq_path = Paths.curves / f"{ticker.lower()}_equity_curve.csv"
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        bt.equity_curve.to_csv(eq_path, index=False)
        bt_metrics_path = Paths.metrics / f"{ticker.lower()}_backtest_metrics.json"
        bt_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        bt_metrics_path.write_text(json.dumps(bt.metrics, indent=2), encoding="utf-8")

        latest = bt.equity_curve.iloc[-1]
        from stock_analyzer.features import FEATURE_COLUMNS

        top_features = infer_feature_importance(model, FEATURE_COLUMNS)
        write_markdown_report(
            path=Paths.markdown_reports / f"{ticker.lower()}_report.md",
            ticker=ticker,
            model_metrics={
                "valid_accuracy": model_metrics["valid"]["accuracy"],
                "valid_roc_auc": model_metrics["valid"]["roc_auc"],
                "test_accuracy": model_metrics["test"]["accuracy"],
                "test_roc_auc": model_metrics["test"]["roc_auc"],
                "best_threshold": model_metrics["best_threshold"],
            },
            backtest_metrics=bt.metrics,
            latest_row=latest,
            top_features=top_features,
            diagnostics={
                "model_quality_flag": model_metrics["model_quality_flag"],
                "walk_forward_avg_accuracy": 0.0,
                "walk_forward_avg_roc_auc": 0.0,
            },
        )

        summary_rows.append(
            {
                "ticker": ticker,
                "test_accuracy": model_metrics["test"]["accuracy"],
                "test_roc_auc": model_metrics["test"]["roc_auc"],
                "best_threshold": best_threshold,
                "strategy_total_return": bt.metrics["strategy_total_return"],
                "buy_hold_total_return": bt.metrics["buy_hold_total_return"],
                "model_quality_flag": model_metrics["model_quality_flag"],
            }
        )
        print(f"Completed {ticker}.")

    if summary_rows:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out, index=False)
        print(f"Wrote summary: {out}")
    else:
        print("No tickers completed.")


if __name__ == "__main__":
    main()
