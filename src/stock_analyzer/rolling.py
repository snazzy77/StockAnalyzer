"""Monthly walk-forward Ridge forecasts using only recently completed outcomes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .regression import select_features


@dataclass
class RecentModel:
    model: object
    mean: pd.Series
    scale: pd.Series
    baseline: float
    audit: dict

    def predict(self, rows):
        if self.model is None:
            return pd.Series(self.baseline, index=rows.index)
        return pd.Series(self.model.predict((rows[self.mean.index] - self.mean) / self.scale), index=rows.index)


def recent_training(frame, asof, quarters):
    """Window is measured by feature-origin dates, not by label completion dates."""
    history = frame.loc[:asof].tail(quarters * 63)
    return history[(history.target_end <= asof) & (history.index < asof)].dropna(subset=["target"])


def fit_recent(frame, asof, quarters, alpha, horizon, max_peers=2, preparation_cache=None, feature_set="baseline"):
    train = recent_training(frame, asof, quarters)
    if len(train) < 40:
        raise ValueError("Fewer than 40 completed recent training outcomes")
    # Preserve the original core and peer budget in both feature experiments.
    core = [c for c in ("own_return_21", "relative_volume_21", "market_return_21",
                         "market_volatility_21", "vix_level") if c in train]
    peer_budget = min(max_peers, 2, max(0, 6 - len(core)))
    if feature_set not in ("baseline", "enhanced"):
        raise ValueError("Unknown feature set")
    if feature_set == "enhanced":
        core += [c for c in train if c.startswith("enh_")]
    cache_key = (asof, quarters, max_peers, feature_set)
    if alpha is None:
        columns = []
    elif preparation_cache is not None and cache_key in preparation_cache:
        columns = preparation_cache[cache_key]
    else:
        subset = train[core + [c for c in train if c.startswith("peer_")] + ["target", "target_end"]]
        columns, _ = select_features(subset, peer_budget)
        if preparation_cache is not None:
            preparation_cache[cache_key] = columns
    data = train.dropna(subset=columns)
    if len(data) < 40 or (alpha is not None and not columns):
        raise ValueError("Insufficient complete recent inputs")
    mean, scale = data[columns].mean(), data[columns].std()
    model = None if alpha is None else Ridge(alpha=alpha).fit((data[columns] - mean) / scale, data.target)
    positions = frame.index.get_indexer(data.index)
    count, last = 0, -horizon
    for pos in positions:
        if pos >= last + horizon:
            count, last = count + 1, pos
    audit = {"asof": str(pd.Timestamp(asof).date()), "quarters": quarters,
             "alpha": alpha, "feature_set": feature_set, "rows": len(data), "first_origin": str(data.index.min().date()),
             "last_origin": str(data.index.max().date()),
             "last_target_end": str(data.target_end.max().date()),
             "nonoverlapping_training_origins": count, "features": columns}
    audit["available_window_origins"] = min(len(frame.loc[:asof]), quarters * 63)
    audit["full_window_available"] = len(frame.loc[:asof]) >= quarters * 63
    return RecentModel(model, mean, scale, float(data.target.mean()), audit)


def walk_forward(frame, dates, quarters, alpha, horizon, max_peers, preparation_cache=None, feature_set="baseline"):
    predictions = pd.Series(np.nan, index=dates, dtype=float)
    audits = []
    for _, month_dates in pd.Series(dates, index=dates).groupby(dates.to_period("M")):
        asof = month_dates.iloc[0]
        try:
            bundle = fit_recent(frame, asof, quarters, alpha, horizon, max_peers, preparation_cache, feature_set)
        except ValueError as exc:
            audits.append({"asof": str(asof.date()), "skipped": str(exc)})
            continue
        rows = frame.loc[month_dates.index].dropna(subset=list(bundle.mean.index))
        predictions.loc[rows.index] = bundle.predict(rows)
        audits.append(bundle.audit)
    return predictions, audits


def score(actual, predicted, baseline):
    error = actual - predicted
    denominator = float(((actual - baseline) ** 2).sum())
    return {"rows": len(actual), "rmse": float(np.sqrt((error ** 2).mean())),
            "mae": float(error.abs().mean()),
            "rank_ic": float(actual.corr(predicted, method="spearman")) if len(actual) >= 3 and actual.nunique() > 1 and predicted.nunique() > 1 else None,
            "baseline_rmse": float(np.sqrt(((actual - baseline) ** 2).mean())),
            "r2_vs_rolling_mean": 1 - float((error ** 2).sum()) / denominator if denominator else None}


def run_rolling(frame, train_end, valid_end, horizon, max_peers=2,
                quarters=(2, 3, 4, 8, 12, 20), alphas=(1.0, 10.0, 100.0), benchmark_quarters=12,
                feature_sets=("baseline", "enhanced"), validation_periods=3):
    if any(not isinstance(q, int) or q < 2 or q > 20 for q in quarters) or not quarters:
        raise ValueError("Training windows must be between 2 and 20 quarters")
    if not alphas or any(a <= 0 for a in alphas) or benchmark_quarters <= 4:
        raise ValueError("Use positive Ridge penalties and a benchmark longer than four quarters")
    if pd.Timestamp(train_end) >= pd.Timestamp(valid_end):
        raise ValueError("train-end must precede valid-end")
    if not feature_sets or any(f not in ("baseline", "enhanced") for f in feature_sets):
        raise ValueError("Use baseline and/or enhanced feature sets")
    if validation_periods < 2:
        raise ValueError("Use at least two validation periods")
    dates = frame.index[frame.index > pd.Timestamp(train_end)]
    configs = {f"ridge_{q}q_a{a:g}" + ("_enhanced" if f == "enhanced" else ""): (q, a, f)
               for q in quarters for a in alphas for f in feature_sets}
    configs["mean_4q"] = (4, None, "baseline")
    configs[f"benchmark_{benchmark_quarters}q"] = (benchmark_quarters, 10.0, "baseline")
    predictions = pd.DataFrame({"actual_return": frame.loc[dates, "target"]})
    audit = {}
    preparation_cache = {}
    for name, (q, alpha, feature_set) in configs.items():
        predictions[name], audit[name] = walk_forward(frame, dates, q, alpha, horizon, max_peers, preparation_cache, feature_set)
    valid_mask = (predictions.index <= pd.Timestamp(valid_end)) & (frame.loc[dates, "target_end"] <= pd.Timestamp(valid_end))
    valid = predictions.loc[valid_mask].dropna()
    test = predictions.loc[predictions.index > pd.Timestamp(valid_end)].dropna()
    if len(valid) < 40 or len(test) < 40:
        raise ValueError("Need 40 common completed validation and test origins; extend history/evaluation dates")
    comparisons = {name: {"validation": score(valid.actual_return, valid[name], valid.mean_4q),
                          "test": score(test.actual_return, test[name], test.mean_4q)} for name in configs}
    # Purge each validation block at its end; all candidates share its dates.
    blocks = []
    for positions in np.array_split(np.arange(len(valid)), validation_periods):
        block = valid.iloc[positions]
        if block.empty:
            raise ValueError("Empty validation period")
        block = block.loc[frame.loc[block.index, "target_end"] <= block.index[-1]]
        if len(block) < 20:
            raise ValueError("Need 20 completed origins per purged validation period; extend validation or reduce periods")
        blocks.append(block)
    for name in configs:
        comparisons[name]["validation_periods"] = [
            {"start": str(b.index[0].date()), "end": str(b.index[-1].date()),
             **score(b.actual_return, b[name], b.mean_4q)} for b in blocks]
        comparisons[name]["selection_rmse"] = float(np.mean([
            item["rmse"] for item in comparisons[name]["validation_periods"]]))
        comparisons[name]["test_years"] = {
            str(year): score(b.actual_return, b[name], b.mean_4q)
            for year, b in test.groupby(test.index.year)}
        keep, last = [], -horizon
        for date in test.index:
            pos = frame.index.get_loc(date)
            if pos >= last + horizon:
                keep.append(date)
                last = pos
        b = test.loc[keep]
        comparisons[name]["test_nonoverlap"] = score(b.actual_return, b[name], b.mean_4q)
    # The duplicate fixed benchmark remains diagnostic; longer searched windows are eligible.
    eligible = [name for name in configs if not name.startswith("benchmark_")]
    chosen = min(eligible, key=lambda name: (comparisons[name]["selection_rmse"], name != "mean_4q", name.endswith("_enhanced")))
    validation_errors = []
    last = -horizon
    for date, row in valid.iterrows():
        pos = frame.index.get_loc(date)
        if pos >= last + horizon:
            validation_errors.append(float(row.actual_return - row[chosen]))
            last = pos
    q, alpha, feature_set = configs[chosen]
    latest = fit_recent(frame, frame.index[-1], q, alpha, horizon, max_peers, feature_set=feature_set)
    latest_rows = frame.iloc[[-1]]
    expected, contributions = None, {}
    if latest_rows[list(latest.mean.index)].notna().all(axis=None):
        expected = float(latest.predict(latest_rows).iloc[0])
        if latest.model is None:
            contributions = {"intercept": latest.baseline}
        else:
            contributions = (((latest_rows[latest.mean.index].iloc[0] - latest.mean) / latest.scale)
                             * latest.model.coef_).to_dict()
            contributions["intercept"] = float(latest.model.intercept_)
    interval = None
    if expected is not None and len(validation_errors) >= 8:
        low, high = np.quantile(validation_errors, [.1, .9])
        interval = [expected + float(low), expected + float(high)]
    coefficients = pd.DataFrame({"feature": list(latest.mean.index) + ["intercept"],
        "ridge_coefficient": list(latest.model.coef_) + [float(latest.model.intercept_)] if latest.model is not None else [latest.baseline]})
    test_skill = comparisons[chosen]["test"]["r2_vs_rolling_mean"]
    report = {"mode": "rolling", "horizon_trading_days": horizon, "selected_variant": chosen,
              "selection": "Mean RMSE across purged validation periods only; fixed benchmark excluded; test never selects configuration",
              "comparisons": comparisons, "latest_fit": latest.audit,
              "validation_period_count": validation_periods,
              "feature_sets": list(feature_sets),
              "evaluation_requested_rows": {"validation": int(valid_mask.sum()), "test": int(((predictions.index > pd.Timestamp(valid_end)) & predictions.actual_return.notna()).sum())},
              "latest_date": str(frame.index[-1].date()), "latest_expected_return": expected,
              "latest_contributions": contributions, "empirical_80pct_error_range": interval,
              "interval_validation_nonoverlap_rows": len(validation_errors),
              "signal_status": "no_useful_signal" if chosen == "mean_4q" or test_skill is None or test_skill <= 0 else "positive_test_skill_unconfirmed",
              "monthly_fit_audit": audit,
              "evaluation_common_rows": {"validation": len(valid), "test": len(test)},
              "inference_note": "Ridge has no ordinary OLS p/t/F statistics. Error range uses non-overlapping validation residuals; coverage is not guaranteed."}
    predictions["evaluation_split"] = "unscored"
    predictions.loc[valid.index, "evaluation_split"] = "validation"
    predictions.loc[test.index, "evaluation_split"] = "test"
    return report, coefficients, predictions, latest


def render_report(report, coefficients):
    lines = [f"# {report['ticker']}: rolling {report['horizon_trading_days']}-day regression", "",
             f"Selected configuration: **{report['selected_variant']}**",
             f"Signal status: **{report['signal_status']}**", "",
             f"Latest input date: {report['latest_date']}",
             f"Estimated return (decimal): {report['latest_expected_return']}",
             f"Empirical 80% error range: {report['empirical_80pct_error_range']}", "",
             "## Latest recent-data refit", ""]
    lines += [f"- {key}: {value}" for key, value in report["latest_fit"].items()]
    lines += ["", "## Walk-forward comparison", "", "| Configuration | Validation RMSE | Test RMSE | Test R² vs rolling mean |",
              "|---|---:|---:|---:|"]
    for name, item in report["comparisons"].items():
        lines.append(f"| {name} | {item['validation']['rmse']:.5f} | {item['test']['rmse']:.5f} | {item['test']['r2_vs_rolling_mean']} |")
    lines += ["", "## Signal diagnostics", "",
              "Selection uses average RMSE across purged validation periods. Test years are diagnostic only.",
              "Rank IC is a time-series Spearman correlation; overlapping outcomes are dependent.", "",
              "| Configuration | Selection RMSE | Test rank IC | Non-overlap rows | Non-overlap RMSE |",
              "|---|---:|---:|---:|---:|"]
    for name, item in report["comparisons"].items():
        nonoverlap = item["test_nonoverlap"]
        lines.append(f"| {name} | {item['selection_rmse']:.5f} | {item['test']['rank_ic']} | {nonoverlap['rows']} | {nonoverlap['rmse']:.5f} |")
    chosen = report["comparisons"][report["selected_variant"]]
    lines += ["", "### Selected model across periods", ""]
    for i, item in enumerate(chosen["validation_periods"], 1):
        lines.append(f"- Validation {i} ({item['start']} to {item['end']}): RMSE {item['rmse']:.5f}; baseline {item['baseline_rmse']:.5f}; rank IC {item['rank_ic']}.")
    for year, item in chosen["test_years"].items():
        lines.append(f"- Test {year}: RMSE {item['rmse']:.5f}; baseline {item['baseline_rmse']:.5f}; rank IC {item['rank_ic']}.")
    lines += ["", "## Latest coefficients", "", "Per training standard deviation; no OLS significance tests apply.", ""]
    lines += [f"- {row.feature}: {row.ridge_coefficient:.6f}" for row in coefficients.itertuples()]
    lines += ["", "## Latest contributions", ""]
    lines += [f"- {key}: {value:.6f}" for key, value in report["latest_contributions"].items()]
    lines += ["", "## Limitations", "", report["inference_note"],
              "Quarterly outcomes overlap heavily; the training-origin count above measures non-overlapping samples, not proof of statistical independence.",
              "Each historical model refits at the first observed trading day of the month after close; the latest model is refitted separately as of its input date.",
              "Candidate universe is fixed from discovery; current Yahoo membership can introduce survivorship bias. SEC candidates remain unverified.",
              "This is forecast evaluation, not an executable portfolio backtest."]
    lines += [f"- {warning}" for warning in report.get("warnings", [])]
    return "\n".join(lines)
