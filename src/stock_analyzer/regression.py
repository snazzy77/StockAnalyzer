"""Return regression with training-only selection and purged chronological evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def features(prices: pd.DataFrame, volume: pd.Series, ticker: str, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    close = prices[ticker]
    # Assemble once: repeated DataFrame insertion fragments large peer universes.
    out = {}
    for window in (5, 21, 63, 252):
        out[f"own_return_{window}"] = close.pct_change(window, fill_method=None)
    out["own_volatility_21"] = close.pct_change(fill_method=None).rolling(21).std()
    out["volume_change_5"] = volume.pct_change(5, fill_method=None)
    out["relative_volume_21"] = volume / volume.rolling(21).mean() - 1
    out["market_return_21"] = prices["SPY"].pct_change(21, fill_method=None)
    out["market_volatility_21"] = prices["SPY"].pct_change(fill_method=None).rolling(21).std()
    if "^VIX" in prices:
        out["vix_level"] = prices["^VIX"]
        out["vix_change_5"] = prices["^VIX"].diff(5)
    for name in prices:
        if name not in (ticker, "SPY", "^VIX"):
            out[f"peer_{name}"] = prices[name].pct_change(21, fill_method=None) - out["market_return_21"]
    out["target"] = close.shift(-horizon) / close - 1
    out["target_end"] = pd.Series(prices.index, index=prices.index).shift(-horizon)
    return pd.DataFrame(out, index=prices.index).replace([np.inf, -np.inf], np.nan)


def purged_split(frame: pd.DataFrame, train_end: str, valid_end: str):
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    if train_end >= valid_end:
        raise ValueError("train_end must precede valid_end")
    train = frame[(frame.index <= train_end) & (frame.target_end <= train_end)]
    valid = frame[(frame.index > train_end) & (frame.index <= valid_end) & (frame.target_end <= valid_end)]
    test = frame[frame.index > valid_end].dropna(subset=["target"])
    return train, valid, test


def select_features(train: pd.DataFrame, max_peers: int = 4):
    # Keep core inputs first; exclude constants, scarce coverage and redundant columns.
    core = [c for c in train if c not in ("target", "target_end") and not c.startswith("peer_")]
    scores = []
    for col in [c for c in train if c.startswith("peer_")]:
        halves = np.array_split(np.arange(len(train)), 2)
        correlations = [train.iloc[ix][col].corr(train.iloc[ix]["target"]) for ix in halves]
        stable = all(np.isfinite(c) for c in correlations) and correlations[0] * correlations[1] > 0
        scores.append({"feature": col, "first_half_correlation": correlations[0],
                       "second_half_correlation": correlations[1],
                       "score": min(abs(c) for c in correlations) if stable else 0.0})
    ranked = sorted(scores, key=lambda row: row["score"], reverse=True)
    candidates = core + [r["feature"] for r in ranked if r["score"] > 0]
    selected = []
    for col in candidates:
        if train[col].notna().mean() < 0.95 or train[col].std() < 1e-10:
            continue
        if col.startswith("peer_") and sum(c.startswith("peer_") for c in selected) >= max_peers:
            continue
        if any(abs(train[col].corr(train[other])) > 0.9 for other in selected):
            continue
        selected.append(col)
    return selected, scores


def fit(train: pd.DataFrame, columns: list[str], horizon: int):
    data = train.dropna(subset=columns + ["target"])
    if len(data) < max(252, 15 * (len(columns) + 1)):
        raise ValueError("Insufficient complete training observations; provide more history or fewer inputs.")
    if not columns:
        raise ValueError("No usable features")
    mean = data[columns].mean()
    scale = data[columns].std()
    x = sm.add_constant((data[columns] - mean) / scale, has_constant="add")
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("Rank-deficient regression; reduce redundant candidates")
    model = sm.OLS(data.target, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon - 1}, use_t=True)
    return model, mean, scale


def predict(bundle, frame: pd.DataFrame):
    model, mean, scale = bundle
    x = sm.add_constant((frame[mean.index] - mean) / scale, has_constant="add")
    return model.predict(x)


def metrics(y: pd.Series, prediction: pd.Series, baseline: float) -> dict:
    if len(y) < 20:
        raise ValueError("Need at least 20 complete evaluation rows per split")
    error = y - prediction
    denominator = float(((y - baseline) ** 2).sum())
    return {"rows": len(y), "rmse": float(np.sqrt((error ** 2).mean())),
            "mae": float(error.abs().mean()),
            "out_of_sample_r2_vs_train_mean": 1 - float((error ** 2).sum()) / denominator if denominator else None,
            "direction_accuracy": float(((prediction > 0) == (y > 0)).mean()),
            "baseline_rmse": float(np.sqrt(((y - baseline) ** 2).mean()))}


def run_regression(frame: pd.DataFrame, train_end: str, valid_end: str, horizon: int,
                   max_peers: int = 4):
    train, valid, test = purged_split(frame, train_end, valid_end)
    # Warm-up is intentional, not missing-provider coverage. Retain long lookbacks.
    train = train.dropna(subset=["own_return_252", "target"])
    if train.empty or valid.empty or test.empty:
        raise ValueError("Empty chronological split; adjust dates or history")
    columns, scores = select_features(train, max_peers)
    core = [c for c in columns if not c.startswith("peer_")]
    own = [c for c in core if c.startswith(("own_", "volume_", "relative_volume_"))]
    variants = {"own_only": own, "market": core, "adjacency": columns}
    # Common evaluation rows make the incremental comparison meaningful.
    valid = valid.dropna(subset=columns + ["target"])
    test = test.dropna(subset=columns + ["target"])
    fitted, comparisons = {}, {}
    for name, cols in variants.items():
        bundle = fit(train, cols, horizon)
        fitted[name] = bundle
        comparisons[name] = {"validation": metrics(valid.target, predict(bundle, valid), float(bundle[0].model.endog.mean()))}
    chosen = min(comparisons, key=lambda name: comparisons[name]["validation"]["rmse"])
    predictions = pd.DataFrame({"actual_return": test.target})
    for name, bundle in fitted.items():
        prediction = predict(bundle, test)
        predictions[name] = prediction
        comparisons[name]["test"] = metrics(test.target, prediction, float(bundle[0].model.endog.mean()))
        # Non-overlapping origin sample is a diagnostic, not a strategy backtest.
        positions = frame.index.get_indexer(test.index)
        keep, last = [], -horizon
        for i, pos in enumerate(positions):
            if pos >= last + horizon:
                keep.append(i)
                last = pos
        comparisons[name]["test_nonoverlap_rows"] = len(keep)
        comparisons[name]["test_nonoverlap_rmse"] = float(np.sqrt(((test.target.iloc[keep] - prediction.iloc[keep]) ** 2).mean()))
    model, mean, scale = fitted[chosen]
    ci = model.conf_int()
    coefficients = pd.DataFrame({"coefficient": model.params, "std_error_hac": model.bse,
                                 "t_value": model.tvalues, "p_value": model.pvalues,
                                 "ci_95_low": ci[0], "ci_95_high": ci[1]})
    # Never silently label an older complete row as the latest prediction.
    latest = frame.iloc[[-1]]
    latest_prediction = None
    contributions = {}
    if latest[list(mean.index)].notna().all(axis=None):
        latest_prediction = float(predict(fitted[chosen], latest).iloc[0])
        contributions = (((latest[mean.index].iloc[0] - mean) / scale) * model.params[mean.index]).to_dict()
        contributions["intercept"] = float(model.params["const"])
    report = {"horizon_trading_days": horizon, "selected_variant": chosen,
              "beats_test_mean_baseline": comparisons[chosen]["test"]["rmse"] < comparisons[chosen]["test"]["baseline_rmse"],
              "features": list(mean.index), "train_rows": int(model.nobs),
              "train_last_target_date": str(train.target_end.max().date()),
              "r_squared": float(model.rsquared), "adjusted_r_squared": float(model.rsquared_adj),
              "f_statistic_hac": float(model.fvalue), "f_p_value_hac": float(model.f_pvalue),
              "hac_lags": horizon - 1, "comparisons": comparisons, "candidate_scores": scores,
              "latest_date": str(latest.index[0].date()), "latest_expected_return": latest_prediction,
              "latest_contributions": contributions,
              "inference_note": "HAC statistics are exploratory after feature selection; no causal or calibrated probability interpretation."}
    return report, coefficients, predictions, fitted[chosen]
