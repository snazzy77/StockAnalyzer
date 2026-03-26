from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import BDay

from .sentiment import compute_sentiment


EVENT_KEYWORDS = {
    "earnings": ["earnings", "eps", "guidance"],
    "product": ["launch", "product", "release", "chip"],
    "legal": ["lawsuit", "probe", "investigation", "fine"],
    "analyst": ["upgrade", "downgrade", "price target", "outperform"],
}


@dataclass
class FeatureConfig:
    horizon_days: int = 5
    prefer_finbert: bool = False


def _detect_event_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    counts: dict[str, int] = {}
    for event_name, words in EVENT_KEYWORDS.items():
        counts[event_name] = int(any(w in lowered for w in words))
    return counts


def _effective_trade_date(ts: pd.Timestamp) -> pd.Timestamp:
    est = ts.tz_convert("America/New_York")
    date = est.normalize()
    # Weekend news belongs to next business day.
    if est.weekday() >= 5:
        date = date + BDay(1)
    # After close news (4pm ET+) belongs to next business day.
    elif est.hour >= 16:
        date = date + BDay(1)
    return pd.Timestamp(date).tz_localize(None)


def build_news_daily_features(news_df: pd.DataFrame, prefer_finbert: bool = False) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "news_count",
                "avg_sentiment",
                "weighted_sentiment",
                "pos_news_count",
                "neg_news_count",
                "neutral_news_count",
            ]
        )

    required = {"published_at", "title", "content"}
    missing = required - set(news_df.columns)
    if missing:
        raise ValueError(f"news dataframe missing columns: {sorted(missing)}")

    temp = news_df.copy()
    temp["published_at"] = pd.to_datetime(temp["published_at"], utc=True, errors="coerce")
    temp = temp.dropna(subset=["published_at"])
    temp["date"] = temp["published_at"].map(_effective_trade_date)
    temp["title"] = temp["title"].fillna("").astype(str).str.strip()
    temp["content"] = temp["content"].fillna("").astype(str).str.strip()
    temp = temp.drop_duplicates(subset=["date", "title", "content"])

    text_series = (temp["title"] + " " + temp["content"]).astype(str)
    temp["sentiment"] = text_series.map(lambda txt: compute_sentiment(txt, prefer_finbert=prefer_finbert))
    temp["is_pos"] = (temp["sentiment"] > 0.1).astype(int)
    temp["is_neg"] = (temp["sentiment"] < -0.1).astype(int)
    temp["is_neutral"] = ((temp["sentiment"] >= -0.1) & (temp["sentiment"] <= 0.1)).astype(int)
    temp["sentiment_abs"] = temp["sentiment"].abs()

    event_rows = text_series.map(_detect_event_counts).apply(pd.Series)
    for col in event_rows.columns:
        temp[col] = event_rows[col]

    grouped = temp.groupby("date", as_index=False).agg(
        news_count=("sentiment", "size"),
        avg_sentiment=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        avg_abs_sentiment=("sentiment_abs", "mean"),
        pos_news_count=("is_pos", "sum"),
        neg_news_count=("is_neg", "sum"),
        neutral_news_count=("is_neutral", "sum"),
        earnings_events=("earnings", "sum"),
        product_events=("product", "sum"),
        legal_events=("legal", "sum"),
        analyst_events=("analyst", "sum"),
    )
    grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
    grouped["weighted_sentiment"] = grouped["avg_sentiment"] * grouped["news_count"].map(
        lambda n: math.log1p(float(n))
    )
    grouped["sentiment_balance"] = grouped["pos_news_count"] - grouped["neg_news_count"]
    grouped["date"] = pd.to_datetime(grouped["date"]).dt.normalize()
    grouped = grouped.sort_values("date").reset_index(drop=True)
    grouped["news_count_3d"] = grouped["news_count"].rolling(3, min_periods=1).sum()
    grouped["news_count_7d"] = grouped["news_count"].rolling(7, min_periods=1).sum()
    grouped["weighted_sentiment_3d"] = grouped["weighted_sentiment"].rolling(3, min_periods=1).mean()
    grouped["weighted_sentiment_7d"] = grouped["weighted_sentiment"].rolling(7, min_periods=1).mean()
    return grouped


def build_price_features(prices_df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame()

    temp = prices_df.copy()
    temp["date"] = pd.to_datetime(temp["date"]).dt.normalize()
    temp = temp.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    close = temp["close"]
    temp["ret_1d"] = close.pct_change(1)
    temp["ret_5d"] = close.pct_change(5)
    temp["ret_20d"] = close.pct_change(20)
    temp["vol_20d"] = temp["ret_1d"].rolling(20).std()
    temp["volume_z20"] = (temp["volume"] - temp["volume"].rolling(20).mean()) / temp["volume"].rolling(20).std()
    temp["ma_5"] = close.rolling(5).mean()
    temp["ma_20"] = close.rolling(20).mean()
    temp["ma_5_20_ratio"] = temp["ma_5"] / temp["ma_20"]
    temp["forward_return"] = close.shift(-horizon_days) / close - 1.0
    temp["target_up"] = (temp["forward_return"] > 0).astype(int)
    return temp


def build_model_dataset(
    prices_df: pd.DataFrame, news_df: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    pf = build_price_features(prices_df, horizon_days=config.horizon_days)
    nf = build_news_daily_features(news_df, prefer_finbert=config.prefer_finbert)
    if nf.empty:
        merged = pf.copy()
        merged["news_count"] = 0.0
        merged["avg_sentiment"] = 0.0
        merged["weighted_sentiment"] = 0.0
        merged["sentiment_std"] = 0.0
        merged["avg_abs_sentiment"] = 0.0
        merged["pos_news_count"] = 0.0
        merged["neg_news_count"] = 0.0
        merged["neutral_news_count"] = 0.0
        merged["sentiment_balance"] = 0.0
        merged["news_count_3d"] = 0.0
        merged["news_count_7d"] = 0.0
        merged["weighted_sentiment_3d"] = 0.0
        merged["weighted_sentiment_7d"] = 0.0
        merged["earnings_events"] = 0.0
        merged["product_events"] = 0.0
        merged["legal_events"] = 0.0
        merged["analyst_events"] = 0.0
    else:
        merged = pf.merge(nf, on="date", how="left")
        fill_cols = [
            "news_count",
            "avg_sentiment",
            "weighted_sentiment",
            "sentiment_std",
            "avg_abs_sentiment",
            "pos_news_count",
            "neg_news_count",
            "neutral_news_count",
            "sentiment_balance",
            "news_count_3d",
            "news_count_7d",
            "weighted_sentiment_3d",
            "weighted_sentiment_7d",
            "earnings_events",
            "product_events",
            "legal_events",
            "analyst_events",
        ]
        merged[fill_cols] = merged[fill_cols].fillna(0.0)
    merged["has_news"] = (merged["news_count"] > 0).astype(int)

    merged = merged.dropna(
        subset=[
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "vol_20d",
            "volume_z20",
            "ma_5_20_ratio",
            "forward_return",
        ]
    ).reset_index(drop=True)
    return merged


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "volume_z20",
    "ma_5_20_ratio",
    "has_news",
    "news_count",
    "news_count_3d",
    "news_count_7d",
    "avg_sentiment",
    "weighted_sentiment",
    "weighted_sentiment_3d",
    "weighted_sentiment_7d",
    "sentiment_std",
    "avg_abs_sentiment",
    "pos_news_count",
    "neg_news_count",
    "neutral_news_count",
    "sentiment_balance",
    "earnings_events",
    "product_events",
    "legal_events",
    "analyst_events",
]
