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

from stock_analyzer.config import Paths
from stock_analyzer.modeling import (
    evaluate,
    evaluate_with_threshold,
    save_model,
    temporal_split,
    train_baseline,
    tune_threshold,
    walk_forward_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline stock direction model.")
    parser.add_argument("--dataset", default=str(Paths.processed_data / "dataset.csv"))
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--model-out", default=str(Paths.models / "baseline_model.joblib"))
    parser.add_argument("--metrics-out", default=str(Paths.metrics / "metrics.json"))
    parser.add_argument("--walk-forward-start-year", type=int, default=2019)
    parser.add_argument("--walk-forward-end-year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset, parse_dates=["date"])
    train_df, valid_df, test_df = temporal_split(df, train_end=args.train_end, valid_end=args.valid_end)

    model = train_baseline(train_df)

    valid_metrics, valid_report = evaluate(model, valid_df)
    test_metrics, test_report = evaluate(model, test_df)
    best_threshold, valid_thresh_metrics = tune_threshold(model, valid_df)
    test_thresh_metrics = evaluate_with_threshold(model, test_df, best_threshold)
    wf = walk_forward_evaluate(
        df=df, start_year=args.walk_forward_start_year, end_year=args.walk_forward_end_year
    )
    quality_flag = "predictive" if float(test_metrics.get("roc_auc", 0.0)) >= 0.55 else "non_predictive"

    save_model(model, Path(args.model_out))

    metrics = {
        "valid": valid_metrics,
        "test": test_metrics,
        "valid_at_best_threshold": valid_thresh_metrics,
        "test_at_best_threshold": test_thresh_metrics,
        "best_threshold": best_threshold,
        "walk_forward": wf,
        "model_quality_flag": quality_flag,
        "split": {
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
        },
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Validation metrics:", valid_metrics)
    print("Test metrics:", test_metrics)
    print("Best threshold (validation):", best_threshold)
    print("Validation @ best threshold:", valid_thresh_metrics)
    print("Test @ best threshold:", test_thresh_metrics)
    print("Walk-forward summary:", {"avg_accuracy": wf.get("avg_accuracy"), "avg_roc_auc": wf.get("avg_roc_auc")})
    print("Model quality flag:", quality_flag)
    print("\nValidation report:\n", valid_report)
    print("\nTest report:\n", test_report)
    print(f"Saved model: {args.model_out}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
