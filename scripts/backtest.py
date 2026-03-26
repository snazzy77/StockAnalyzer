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

from stock_analyzer.backtest import run_threshold_strategy
from stock_analyzer.config import Paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple threshold backtest.")
    parser.add_argument("--dataset", default=str(Paths.processed_data / "dataset.csv"))
    parser.add_argument("--model", default=str(Paths.models / "baseline_model.joblib"))
    parser.add_argument("--model-metrics", default=str(Paths.metrics / "metrics.json"))
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--prob-threshold", type=float, default=None)
    parser.add_argument("--trade-cost-bps", type=float, default=5.0)
    parser.add_argument("--equity-out", default=str(Paths.curves / "equity_curve.csv"))
    parser.add_argument("--metrics-out", default=str(Paths.metrics / "backtest_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset, parse_dates=["date"])
    df = df[df["date"] >= pd.to_datetime(args.start_date)].copy()
    if df.empty:
        raise ValueError("No rows left for backtest after start-date filter.")

    model = joblib.load(args.model)
    threshold = args.prob_threshold
    if threshold is None:
        metrics_path = Path(args.model_metrics)
        if metrics_path.exists():
            try:
                threshold = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("best_threshold", 0.55))
            except Exception:
                threshold = 0.55
        else:
            threshold = 0.55
    result = run_threshold_strategy(
        df=df, model=model, prob_threshold=threshold, trade_cost_bps=args.trade_cost_bps
    )
    Path(args.equity_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(args.equity_out, index=False)
    Path(args.metrics_out).write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    print("Backtest metrics:", result.metrics)
    print("Threshold used:", threshold)
    print(f"Saved equity curve: {args.equity_out}")
    print(f"Saved metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
