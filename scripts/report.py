from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stock_analyzer.config import Paths
from stock_analyzer.features import FEATURE_COLUMNS
from stock_analyzer.reporting import infer_feature_importance, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown stock report.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--equity-curve", default=str(Paths.curves / "equity_curve.csv"))
    parser.add_argument("--model", default=str(Paths.models / "baseline_model.joblib"))
    parser.add_argument("--model-metrics", default=str(Paths.metrics / "metrics.json"))
    parser.add_argument("--backtest-metrics", default=str(Paths.metrics / "backtest_metrics.json"))
    parser.add_argument("--out", default=str(Paths.markdown_reports / "report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    equity_df = pd.read_csv(args.equity_curve, parse_dates=["date"])
    if equity_df.empty:
        raise ValueError("Equity curve is empty; run backtest first.")
    latest = equity_df.iloc[-1]

    model = joblib.load(args.model)
    feature_importance = infer_feature_importance(model, FEATURE_COLUMNS)

    model_metrics_json = json.loads(Path(args.model_metrics).read_text(encoding="utf-8"))
    model_metrics = {
        "valid_accuracy": model_metrics_json.get("valid", {}).get("accuracy", 0.0),
        "valid_roc_auc": model_metrics_json.get("valid", {}).get("roc_auc", 0.0),
        "test_accuracy": model_metrics_json.get("test", {}).get("accuracy", 0.0),
        "test_roc_auc": model_metrics_json.get("test", {}).get("roc_auc", 0.0),
        "best_threshold": model_metrics_json.get("best_threshold", 0.55),
    }
    backtest_metrics = json.loads(Path(args.backtest_metrics).read_text(encoding="utf-8"))
    diagnostics = {
        "model_quality_flag": str(model_metrics_json.get("model_quality_flag", "unknown")),
        "walk_forward_avg_accuracy": float(model_metrics_json.get("walk_forward", {}).get("avg_accuracy", 0.0)),
        "walk_forward_avg_roc_auc": float(model_metrics_json.get("walk_forward", {}).get("avg_roc_auc", 0.0)),
    }

    write_markdown_report(
        path=Path(args.out),
        ticker=args.ticker.upper(),
        model_metrics=model_metrics,
        backtest_metrics=backtest_metrics,
        latest_row=latest,
        top_features=feature_importance,
        diagnostics=diagnostics,
    )

    print(f"Report written: {args.out}")


if __name__ == "__main__":
    main()
