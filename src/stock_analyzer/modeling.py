from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from .features import FEATURE_COLUMNS


@dataclass
class TrainResult:
    model: Any
    metrics: dict[str, float]
    report_text: str


def temporal_split(
    df: pd.DataFrame, train_end: str, valid_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    train = dfx[dfx["date"] <= pd.to_datetime(train_end)]
    valid = dfx[(dfx["date"] > pd.to_datetime(train_end)) & (dfx["date"] <= pd.to_datetime(valid_end))]
    test = dfx[dfx["date"] > pd.to_datetime(valid_end)]
    if train.empty or valid.empty or test.empty:
        raise ValueError("Temporal split produced an empty split. Adjust date boundaries.")
    return train, valid, test


def train_baseline(train_df: pd.DataFrame) -> Any:
    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    except Exception:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(train_df[FEATURE_COLUMNS], train_df["target_up"])
    return model


def evaluate(model: Any, df: pd.DataFrame) -> tuple[dict[str, float], str]:
    x = df[FEATURE_COLUMNS]
    y = df["target_up"].to_numpy()
    preds = model.predict(x)
    try:
        probs = model.predict_proba(x)[:, 1]
    except Exception:
        probs = preds.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
    }
    report = classification_report(y, preds, digits=4)
    return metrics, report


def evaluate_with_threshold(model: Any, df: pd.DataFrame, threshold: float) -> dict[str, float]:
    x = df[FEATURE_COLUMNS]
    y = df["target_up"].to_numpy()
    probs = model.predict_proba(x)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan"),
        "signal_rate": float(preds.mean()),
    }


def tune_threshold(model: Any, df: pd.DataFrame, min_thr: float = 0.50, max_thr: float = 0.70) -> tuple[float, dict]:
    best_thr = min_thr
    best_metrics = evaluate_with_threshold(model, df, min_thr)
    best_score = best_metrics["accuracy"]
    for thr in np.arange(min_thr, max_thr + 1e-9, 0.01):
        m = evaluate_with_threshold(model, df, float(round(thr, 2)))
        if m["accuracy"] > best_score:
            best_score = m["accuracy"]
            best_thr = float(round(thr, 2))
            best_metrics = m
    return best_thr, best_metrics


def walk_forward_slices(
    df: pd.DataFrame, start_year: int, end_year: int, min_train_years: int = 3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx["year"] = dfx["date"].dt.year
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_year in range(start_year + min_train_years, end_year + 1):
        train = dfx[dfx["year"] < test_year].copy()
        test = dfx[dfx["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        windows.append((train, test))
    return windows


def walk_forward_evaluate(df: pd.DataFrame, start_year: int, end_year: int) -> dict[str, Any]:
    windows = walk_forward_slices(df, start_year=start_year, end_year=end_year)
    if not windows:
        return {"windows": [], "avg_accuracy": float("nan"), "avg_roc_auc": float("nan")}

    rows = []
    for train_df, test_df in windows:
        model = train_baseline(train_df)
        metrics, _ = evaluate(model, test_df)
        year = int(pd.to_datetime(test_df["date"]).dt.year.iloc[0])
        rows.append({"year": year, **metrics, "rows": int(len(test_df))})

    avg_acc = float(np.nanmean([r["accuracy"] for r in rows])) if rows else float("nan")
    avg_auc = float(np.nanmean([r["roc_auc"] for r in rows])) if rows else float("nan")
    return {"windows": rows, "avg_accuracy": avg_acc, "avg_roc_auc": avg_auc}


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
