from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_price_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf  # Imported lazily to keep package import light.

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker} between {start_date} and {end_date}")

    # yfinance may return MultiIndex columns depending on version/config.
    if isinstance(data.columns, pd.MultiIndex):
        # Most common shape for single ticker is (field, ticker). Extract the field level.
        if ticker.upper() in data.columns.get_level_values(-1):
            data = data.xs(ticker.upper(), axis=1, level=-1, drop_level=True)
        else:
            data.columns = [
                "_".join(str(part) for part in col if part not in ("", None)).strip("_")
                for col in data.columns
            ]

    data = data.reset_index()
    data.columns = [str(c).strip().lower() for c in data.columns]

    rename_map = {
        "date": "date",
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    if "close" not in data.columns and "adj_close" in data.columns:
        data["close"] = data["adj_close"]

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            f"Price dataframe missing required columns {missing}. Found columns: {list(data.columns)}"
        )
    return data[required]


def load_news_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["published_at", "title", "content", "source", "ticker"])
    df = pd.read_csv(path)
    for col in ["published_at", "title", "content"]:
        if col not in df.columns:
            raise ValueError(f"news csv missing required column: {col}")
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    df.to_csv(path, index=False)
