from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from stock_analyzer.config import Defaults, Paths
from stock_analyzer.data import fetch_price_history, load_news_csv, save_dataframe
from stock_analyzer.features import FeatureConfig, build_model_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready stock + news dataset.")
    parser.add_argument("--ticker", default=Defaults.ticker)
    parser.add_argument("--start-date", default=Defaults.start_date)
    parser.add_argument("--end-date", default=Defaults.end_date)
    parser.add_argument("--horizon-days", type=int, default=Defaults.horizon_days)
    parser.add_argument("--news-csv", default=str(Paths.raw_data / "news.csv"))
    parser.add_argument("--prefer-finbert", action="store_true")
    parser.add_argument("--out", default=str(Paths.processed_data / "dataset.csv"))
    parser.add_argument("--stats-out", default=str(Paths.processed_data / "dataset_stats.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices_df = fetch_price_history(args.ticker, args.start_date, args.end_date)
    news_df = load_news_csv(Path(args.news_csv))
    if "ticker" in news_df.columns:
        news_df = news_df[news_df["ticker"].fillna("").str.upper().eq(args.ticker.upper()) | news_df["ticker"].isna()]

    dataset = build_model_dataset(
        prices_df,
        news_df,
        config=FeatureConfig(horizon_days=args.horizon_days, prefer_finbert=args.prefer_finbert),
    )
    dataset["ticker"] = args.ticker.upper()

    save_dataframe(prices_df, Paths.raw_data / f"{args.ticker.upper()}_prices.csv")
    save_dataframe(dataset, Path(args.out))

    stats = {
        "rows": int(len(dataset)),
        "date_min": str(dataset["date"].min()) if not dataset.empty else None,
        "date_max": str(dataset["date"].max()) if not dataset.empty else None,
        "news_days": int((dataset["news_count"] > 0).sum()) if "news_count" in dataset.columns else 0,
        "news_day_ratio": float((dataset["news_count"] > 0).mean()) if "news_count" in dataset.columns else 0.0,
        "avg_news_count_per_day": float(dataset["news_count"].mean()) if "news_count" in dataset.columns else 0.0,
        "avg_sentiment_abs": float(dataset["avg_sentiment"].abs().mean()) if "avg_sentiment" in dataset.columns else 0.0,
    }
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Prepared dataset rows: {len(dataset)}")
    print(f"Wrote: {Path(args.out)}")
    print(f"Wrote stats: {Path(args.stats_out)}")
    if stats["news_day_ratio"] < 0.15:
        print("WARNING: Very low news coverage ratio. News features may have near-zero impact.")
    if stats["avg_sentiment_abs"] < 0.03:
        print("WARNING: Sentiment signal is weak/flat. Consider richer text sources or FinBERT.")


if __name__ == "__main__":
    main()
