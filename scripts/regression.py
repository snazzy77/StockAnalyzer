"""Yahoo + SEC research pipeline. Run --help for options."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stock_analyzer.relationships import discover
from stock_analyzer.regression import features, run_regression
from stock_analyzer.rolling import run_rolling, render_report
from stock_analyzer.reactions import analyze as analyze_reactions, render as render_reactions


def get_sec_user_agent():
    """Environment overrides the optional, git-ignored project-local contact."""
    configured = os.getenv("SEC_USER_AGENT")
    if configured:
        return configured
    path = ROOT / "sec.local.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("SEC_USER_AGENT")
    return None


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def load_history(symbol, start, end, cache, offline=False):
    path = cache / f"{symbol.replace('^', '_')}_{start}_{end}.csv"
    if path.exists():
        data = pd.read_csv(path, index_col=0, parse_dates=True)
    elif offline:
        raise ValueError(f"Missing cached history: {path}")
    else:
        import yfinance as yf
        # This installed yfinance version uses pandas methods being deprecated.
        # Quiet that exact third-party warning only; retain data/runtime warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"The inplace keyword in DataFrame\..* is deprecated.*",
                category=getattr(pd.errors, "Pandas4Warning", FutureWarning),
                module=r"yfinance\.utils",
            )
            data = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No history for {symbol}")
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        data = data[["Close", "Volume"]]
        data.to_csv(path)
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated()].sort_index()
    data = data[(data.index >= pd.Timestamp(start)) & (data.index < pd.Timestamp(end))]
    if data.empty or (data.Close <= 0).any():
        raise ValueError(f"Invalid history for {symbol}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--mode", choices=["rolling", "static", "reactions"], default="rolling")
    parser.add_argument("--analysis-quarters", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--jump-threshold", type=float, default=.03)
    parser.add_argument("--reaction-horizons", type=int, nargs="+", default=[1, 5, 21])
    parser.add_argument("--forecast-days", type=int, default=5)
    parser.add_argument("--training-quarters", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--benchmark-quarters", type=int, default=12)
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).date().isoformat(), help="Exclusive")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--valid-end", default="2023-12-31")
    parser.add_argument("--horizons", type=int, nargs="+", default=[21, 63])
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--max-peers", type=int, default=4)
    parser.add_argument("--no-yahoo-discovery", action="store_true")
    parser.add_argument("--no-sec", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use saved candidate snapshot and cached prices only")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "regression")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "regression")
    args = parser.parse_args()
    if any(q not in (2, 3, 4) for q in args.training_quarters) or any(a <= 0 for a in args.ridge_alphas) or args.benchmark_quarters <= 4:
        parser.error("Use training quarters 2/3/4, positive Ridge alphas, and benchmark quarters > 4")
    if any(h < 1 for h in args.horizons) or args.max_peers < 0:
        parser.error("Horizons must be positive and max-peers nonnegative")
    if not 0 < args.jump_threshold < 1 or args.forecast_days < 1 or any(h < 1 for h in args.reaction_horizons):
        parser.error("Use positive event/forecast horizons and a jump threshold between 0 and 1")
    if not (args.start_date < args.train_end < args.valid_end < args.end_date):
        parser.error("Require start-date < train-end < valid-end < end-date")
    ticker = args.ticker.upper()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        import yfinance as yf
        yf.set_tz_cache_location(str(args.cache_dir / "yahoo"))
    out = args.out_dir / ticker.lower()
    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / "candidates.json"
    if args.offline:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot["train_end"] != args.train_end:
            raise ValueError("Offline candidate snapshot cutoff does not match train-end")
        candidates, warnings = snapshot["candidates"], snapshot["warnings"]
    else:
        print(f"Discovering comparison candidates for {ticker}...", flush=True)
        candidates, warnings = discover(ticker, args.train_end,
            None if args.no_sec else get_sec_user_agent(),
            args.cache_dir / "sec", args.candidates, not args.no_yahoo_discovery)
        snapshot_path.write_text(json.dumps({"train_end": args.train_end, "candidates": candidates,
                                            "warnings": warnings}, indent=2), encoding="utf-8")
    # Keep shared discovery at the ticker root and artifacts grouped by mode.
    out = out / args.mode
    out.mkdir(parents=True, exist_ok=True)
    histories = {}
    symbols = list(dict.fromkeys([ticker, "SPY", "^VIX"] + [r["ticker"] for r in candidates]))
    for symbol in symbols:
        print(f"Loading {symbol} history...", flush=True)
        try:
            histories[symbol] = load_history(symbol, args.start_date, args.end_date, args.cache_dir, args.offline)
        except Exception as exc:
            if symbol in (ticker, "SPY"):
                raise
            warnings.append(f"Skipped {symbol}: {exc}")
    prices = pd.DataFrame({s: df.Close for s, df in histories.items()}).reindex(histories[ticker].index)
    if args.mode == "reactions":
        print("Analyzing related-stock jumps, dives and conditional forecasts...", flush=True)
        report, events, associations, comparisons, predictions, model = analyze_reactions(
            prices, ticker, candidates, args.analysis_quarters, args.jump_threshold,
            tuple(args.reaction_horizons), args.forecast_days)
        report["warnings"] = warnings
        (out / "reactions.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
        for name, data in (("events", events), ("associations", associations), ("model_comparison", comparisons)):
            data.to_csv(out / f"reactions_{name}.csv", index=False)
        predictions.to_csv(out / "reactions_predictions.csv", index_label="date")
        joblib.dump(model, out / "reactions_model.joblib")
        (out / "reactions.md").write_text(render_reactions(report, events, associations, comparisons), encoding="utf-8")
        print(f"Wrote {out / 'reactions.md'}", flush=True)
        for warning in warnings:
            print("Warning:", warning)
        return
    # No forward/back filling of missing prices; complete observations only.
    for horizon in args.horizons:
        print(f"Fitting {horizon}-trading-day regression...", flush=True)
        frame = features(prices, histories[ticker].Volume, ticker, horizon)
        if args.mode == "rolling":
            report, coefficients, predictions, bundle = run_rolling(
                frame, args.train_end, args.valid_end, horizon, args.max_peers,
                tuple(args.training_quarters), tuple(args.ridge_alphas), args.benchmark_quarters)
        else:
            report, coefficients, predictions, bundle = run_regression(
                frame, args.train_end, args.valid_end, horizon, args.max_peers)
        report["warnings"] = warnings
        report["ticker"] = ticker
        report["split"] = {"start_date": args.start_date, "train_end": args.train_end,
                           "valid_end": args.valid_end, "end_date_exclusive": args.end_date}
        stem = f"rolling_{horizon}d" if args.mode == "rolling" else f"{horizon}d"
        prefix = out / stem
        prefix.with_suffix(".json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
        coefficients.to_csv(out / f"{stem}_coefficients.csv", index_label="feature_index")
        predictions.to_csv(out / f"{stem}_predictions.csv", index_label="date")
        joblib.dump(bundle, out / f"{stem}_model.joblib")
        if args.mode == "rolling":
            prefix.with_suffix(".md").write_text(render_report(report, coefficients), encoding="utf-8")
            print(f"Wrote {prefix.with_suffix('.md')}", flush=True)
            continue
        lines = [f"# {ticker}: {horizon}-trading-day return regression", "",
                 f"Selected on validation RMSE: **{report['selected_variant']}**", "",
                 ("Selected model beat the training-mean baseline on the test period; future performance remains unverified."
                  if report["beats_test_mean_baseline"] else
                  "**Selected model did not beat the training-mean baseline on the test period. Treat its forecast as unvalidated.**"), "",
                 f"Latest input date: {report['latest_date']}",
                 f"Estimated return (decimal, not an up probability): {report['latest_expected_return']}", "",
                 f"Training R²: {report['r_squared']:.4f}; adjusted R²: {report['adjusted_r_squared']:.4f}",
                 f"HAC F-statistic: {report['f_statistic_hac']:.4f}; p-value: {report['f_p_value_hac']:.4g}", "",
                 "## Model comparison", "", "| Model | Validation RMSE | Test RMSE | Test R² vs training mean |",
                 "|---|---:|---:|---:|"]
        for name, values in report["comparisons"].items():
            lines.append(f"| {name} | {values['validation']['rmse']:.4f} | {values['test']['rmse']:.4f} | {values['test']['out_of_sample_r2_vs_train_mean']} |")
        lines += ["", "## Coefficients", "", "Coefficients measure return change per training-standard-deviation increase in the input.", "",
                  "| Feature | Coefficient | HAC t | p | 95% interval |", "|---|---:|---:|---:|---|"]
        for name, row in coefficients.iterrows():
            lines.append(f"| {name} | {row.coefficient:.5f} | {row.t_value:.3f} | {row.p_value:.4g} | [{row.ci_95_low:.5f}, {row.ci_95_high:.5f}] |")
        lines += ["", "## Latest prediction contributions", "", "Contributions sum to predicted return; these are model arithmetic, not causal explanations.", ""]
        lines += [f"- {name}: {value:.6f}" for name, value in report["latest_contributions"].items()]
        lines += ["", "## Limitations", "", report["inference_note"],
                  "Daily horizon returns overlap. HAC and non-overlapping diagnostics reduce, but do not eliminate, inference uncertainty.",
                  "SEC mentions are unverified candidates; see ../candidates.json for passages and dates.",
                  "This evaluates return forecasts, not executable trading returns. No transaction-cost backtest is implied.",
                  "Model remains fitted on the training period; latest forecasts are not a full-history production refit."]
        lines += [f"- {warning}" for warning in warnings]
        prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {prefix.with_suffix('.md')}")
    for warning in warnings:
        print("Warning:", warning)


if __name__ == "__main__":
    main()
