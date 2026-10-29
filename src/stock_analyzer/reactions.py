"""Descriptive peer-event responses and a separately evaluated conditional forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def event_responses(prices, ticker, peers, quarters=4, jump=.03, horizons=(1, 5, 21)):
    returns = prices.pct_change(fill_method=None)
    recent_dates = prices.index[-quarters * 63:]
    rows = []
    for peer in peers:
        for horizon in horizons:
            future = prices[ticker].shift(-horizon) / prices[ticker] - 1
            market = prices.SPY.shift(-horizon) / prices.SPY - 1
            eligible = pd.DataFrame({"move": returns[peer], "reaction": future,
                                     "market_excess": future - market, "same_day": returns[ticker]}).loc[recent_dates].dropna()
            selected, last = [], -horizon - 1
            # One chronology across BOTH signs; keep the first event, then skip overlaps.
            for date, row in eligible.iterrows():
                position = prices.index.get_loc(date)
                if abs(row.move) >= jump and position >= last + horizon:
                    selected.append(date)
                    last = position
            for direction in ("jump", "dive"):
                sample = eligible.loc[selected]
                sample = sample[sample.move >= jump] if direction == "jump" else sample[sample.move <= -jump]
                n = len(sample)
                rows.append({"peer": peer, "direction": direction, "horizon": horizon, "events": n,
                             "mean_peer_move": sample.move.mean(), "mean_same_day_nvda_return": sample.same_day.mean(),
                             "mean_return": sample.reaction.mean(), "median_return": sample.reaction.median(),
                             "up_frequency": (sample.reaction > 0).mean() if n else np.nan,
                             "p10_return": sample.reaction.quantile(.1), "p90_return": sample.reaction.quantile(.9),
                             "usual_return": eligible.reaction.mean(),
                             "difference_from_usual": sample.reaction.mean() - eligible.reaction.mean(),
                             "mean_market_excess": sample.market_excess.mean(),
                             "sample_status": "small_sample" if n < 10 else "descriptive_only",
                             "event_dates": ";".join(str(d.date()) for d in sample.index)})
    return pd.DataFrame(rows)


def association_summary(prices, ticker, peers, quarters):
    daily = prices.pct_change(fill_method=None).tail(quarters * 63)
    rows = []
    for peer in peers:
        columns = list(dict.fromkeys([ticker, peer, "SPY"] + (["SOXX"] if "SOXX" in prices else [])))
        data = daily[columns].dropna()
        if len(data) < 40:
            continue
        controls = [c for c in ("SPY", "SOXX") if c in data and c not in (ticker, peer)]
        x = np.column_stack([np.ones(len(data)), data[controls]])
        own = data[ticker].to_numpy()
        other = data[peer].to_numpy()
        own_residual = own - x @ np.linalg.lstsq(x, own, rcond=None)[0]
        peer_residual = other - x @ np.linalg.lstsq(x, other, rcond=None)[0]
        partial = np.corrcoef(own_residual, peer_residual)[0, 1] if np.std(peer_residual) > 1e-10 else np.nan
        rows.append({"peer": peer, "days": len(data), "same_day_correlation": data[ticker].corr(data[peer]),
                     "market_sector_adjusted_correlation": partial,
                     "controls": ",".join(controls),
                     "recent_return": prices[peer].iloc[-1] / prices[peer].iloc[-min(len(prices), quarters * 63)] - 1,
                     "latest_daily_move": daily[peer].iloc[-1]})
    return pd.DataFrame(rows)


def _fit(data, columns):
    clean = data.dropna(subset=columns + ["target"])
    if len(clean) < 50:
        raise ValueError("Need at least 50 complete training outcomes")
    columns = [c for c in columns if clean[c].std() > 1e-10]
    mean, scale = clean[columns].mean(), clean[columns].std()
    estimator = Ridge(alpha=10).fit((clean[columns] - mean) / scale, clean.target) if columns else None
    return {"model": estimator, "mean": mean, "scale": scale, "baseline": float(clean.target.mean()),
            "rows": len(clean), "first_origin": str(clean.index[0].date()),
            "last_origin": str(clean.index[-1].date()), "last_target_end": str(clean.target_end.max().date())}


def _predict(bundle, data):
    if bundle["model"] is None:
        return pd.Series(bundle["baseline"], index=data.index)
    columns = bundle["mean"].index
    return pd.Series(bundle["model"].predict((data[columns] - bundle["mean"]) / bundle["scale"]), index=data.index)


def conditional_forecast(prices, ticker, peers, quarters=4, horizon=5):
    daily = prices.pct_change(fill_method=None)
    inputs = {"own_daily_return": daily[ticker], "own_5d_return": prices[ticker].pct_change(5, fill_method=None),
              "market_daily_return": daily.SPY}
    if "SOXX" in prices and ticker != "SOXX":
        inputs["sector_daily_return"] = daily.SOXX
    core = list(inputs)
    for peer in peers:
        if peer not in ("SPY", "SOXX"):
            inputs[f"peer_{peer}"] = daily[peer]
    inputs["target"] = prices[ticker].shift(-horizon) / prices[ticker] - 1
    inputs["target_end"] = pd.Series(prices.index, index=prices.index).shift(-horizon)
    frame = pd.DataFrame(inputs).tail(quarters * 63).replace([np.inf, -np.inf], np.nan)
    n = len(frame)
    first_cut, second_cut = frame.index[int(n * .6) - 1], frame.index[int(n * .8) - 1]
    train = frame[(frame.index <= first_cut) & (frame.target_end <= first_cut)].dropna(subset=core + ["target"])
    valid = frame[(frame.index > first_cut) & (frame.index <= second_cut) & (frame.target_end <= second_cut)].dropna(subset=core + ["target"])
    test = frame[frame.index > second_cut].dropna(subset=core + ["target"])
    if len(train) < 50 or len(valid) < 10 or len(test) < 10:
        raise ValueError("Insufficient completed observations for purged 60/20/20 forecast evaluation")
    # Require full coverage on comparison dates so each added peer faces the same test.
    usable = [c for c in inputs if c.startswith("peer_") and train[c].notna().all()
              and valid[c].notna().all() and test[c].notna().all() and train[c].std() > 1e-10]
    configurations = {"mean": [], "market_and_own": core}
    configurations.update({c: core + [c] for c in usable})
    bundles = {name: _fit(train, cols) for name, cols in configurations.items()}
    baseline_test = _predict(bundles["market_and_own"], test)
    evaluations = []
    predictions = pd.DataFrame({"actual_return": test.target})
    base_mse = float(((test.target - baseline_test) ** 2).mean())
    for name, bundle in bundles.items():
        vp, tp = _predict(bundle, valid), _predict(bundle, test)
        predictions[name] = tp
        mse = float(((test.target - tp) ** 2).mean())
        evaluations.append({"configuration": name, "validation_rmse": float(np.sqrt(((valid.target - vp) ** 2).mean())),
                            "test_rmse": float(np.sqrt(mse)), "test_rows": len(test),
                            "test_mse_improvement_vs_market_and_own": 1 - mse / base_mse if base_mse > 0 else None})
    comparison = pd.DataFrame(evaluations)
    chosen = comparison.sort_values("validation_rmse", kind="stable").iloc[0].configuration
    # Refit only AFTER validation selection. The last incomplete targets stay excluded.
    completed = frame[frame.target_end <= frame.index[-1]]
    latest = _fit(completed, configurations[chosen])
    recent = frame.iloc[[-1]]
    expected, contributions, factors = None, {}, []
    if recent[list(latest["mean"].index)].notna().all(axis=None):
        expected = float(_predict(latest, recent).iloc[0])
        if latest["model"] is None:
            contributions = {"intercept": latest["baseline"]}
        else:
            coef = pd.Series(latest["model"].coef_, index=latest["mean"].index)
            contributions = (((recent[coef.index].iloc[0] - latest["mean"]) / latest["scale"]) * coef).to_dict()
            contributions["intercept"] = float(latest["model"].intercept_)
            for name in coef.index:
                factors.append({"factor": name, "current_value": float(recent[name].iloc[0]),
                                "contribution": contributions[name],
                                "forecast_change_for_plus_1pp": float(.01 * coef[name] / latest["scale"][name])})
    selected = comparison[comparison.configuration == chosen].iloc[0]
    mean_mse = float(((test.target - predictions["mean"]) ** 2).mean())
    selected_mse = float(((test.target - predictions[chosen]) ** 2).mean())
    model_has_skill = chosen != "mean" and selected_mse < mean_mse
    peer_has_skill = chosen.startswith("peer_") and model_has_skill and selected.test_mse_improvement_vs_market_and_own > 0
    report = {"forecast_horizon": horizon, "selected_configuration": chosen,
              "latest_expected_return": expected, "latest_contributions": contributions, "factors": factors,
              "status": "exploratory_test_improvement" if model_has_skill else "no_demonstrated_predictive_edge",
              "selected_peer_improved_test_error": bool(peer_has_skill),
              "evaluation": {"train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
                             "train_cutoff": str(first_cut.date()), "validation_cutoff": str(second_cut.date())},
              "latest_fit": {k: v for k, v in latest.items() if k not in ("model", "mean", "scale", "baseline")},
              "skipped_peer_features": [c for c in inputs if c.startswith("peer_") and c not in usable]}
    return report, comparison, predictions, latest


def analyze(prices, ticker, candidates, quarters=4, jump=.03, horizons=(1, 5, 21), forecast_horizon=5):
    if quarters not in (2, 3, 4) or not 0 < jump < 1 or any(h < 1 for h in horizons) or forecast_horizon < 1:
        raise ValueError("Use 2–4 quarters, a jump fraction between 0 and 1, and positive horizons")
    sources = {}
    for row in candidates:
        sources.setdefault(row["ticker"], set()).add(row["source"])
    # SEC mentions alone are too ambiguous to label companies as genuine peers.
    peers = [p for p in prices if p not in (ticker, "SPY", "^VIX")
             and sources.get(p, set()) & {"manual", "yahoo_industry", "benchmark"}]
    if not peers:
        raise ValueError("No usable comparison candidates; supply --candidates or enable Yahoo discovery")
    events = event_responses(prices, ticker, peers, quarters, jump, horizons)
    associations = association_summary(prices, ticker, peers, quarters)
    forecast, comparisons, predictions, model = conditional_forecast(prices, ticker, peers, quarters, forecast_horizon)
    report = {"ticker": ticker, "asof": str(prices.index[-1].date()), "quarters": quarters,
              "target_recent_return": float(prices[ticker].iloc[-1] / prices[ticker].iloc[-min(len(prices), quarters * 63)] - 1),
              "event_threshold": jump, "peers": peers, "peer_sources": {p: sorted(sources[p]) for p in peers},
              "excluded_sec_only": sorted(p for p, s in sources.items() if s == {"sec_mention"}),
              "forecast": forecast}
    return report, events, associations, comparisons, predictions, model


def render(report, events, associations, comparisons):
    ticker, f = report["ticker"], report["forecast"]
    def pct(value):
        return "unavailable" if value is None or pd.isna(value) else f"{value * 100:+.2f}%"
    lines = [f"# {ticker}: related-stock reactions and conditional forecast", "",
             f"As of {report['asof']}; descriptive window: last {report['quarters']} quarters (63 trading days each).", "",
             f"{ticker}'s adjusted return over that window: **{pct(report['target_recent_return'])}**.", "",
             "## Conclusion", "",
             f"Next {f['forecast_horizon']} trading days: **{pct(f['latest_expected_return'])}** estimated return.",
             f"Selected configuration: **{f['selected_configuration']}**; status: **{f['status']}**.",
             ("The selected peer model improved on both the mean and market/own-history baselines in this small test. This is preliminary association, not causal influence."
              if f["selected_peer_improved_test_error"] else "No selected peer model demonstrated incremental test improvement over both baselines. Related-stock moves should not yet be treated as reliable drivers."), "",
             "## How closely do they move together?", "",
             "Same-day correlation describes co-movement, not a forecast. Adjusted correlation removes linear SPY and, where available, SOXX effects.", "",
             "| Comparison | Source | Days | Same-day correlation | Adjusted correlation | Recent return | Latest daily move |",
             "|---|---|---:|---:|---:|---:|---:|"]
    if not associations.empty:
        order = associations.assign(strength=associations.market_sector_adjusted_correlation.abs()).sort_values("strength", ascending=False).head(15)
        for r in order.itertuples():
            lines.append(f"| {r.peer} | {', '.join(report['peer_sources'][r.peer])} | {r.days} | {r.same_day_correlation:.3f} | {r.market_sector_adjusted_correlation:.3f} | {pct(r.recent_return)} | {pct(r.latest_daily_move)} |")
    lines += ["", "## Historical jumps and dives", "",
              f"Event = peer's daily adjusted return at least +{100 * report['event_threshold']:.1f}% or at most -{100 * report['event_threshold']:.1f}%.",
              "NVDA response starts at the event-day close; it excludes NVDA's same-day move. Within each peer/horizon, overlapping event windows are skipped across both signs.",
              "Displayed rows prioritize event counts, not large returns. All peers/horizons and event dates are in reactions_events.csv.", "",
              "| Peer | Event | Days ahead | Events | NVDA mean | Median | Up frequency | Usual mean | Difference (pp) | Mean minus SPY |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    event_display = events[events.events > 0].sort_values(["events", "peer"], ascending=[False, True]).groupby("horizon", sort=True).head(10).sort_values(["horizon", "events"], ascending=[True, False])
    for r in event_display.itertuples():
        lines.append(f"| {r.peer} | {r.direction} | {r.horizon} | {r.events}{'*' if r.events < 10 else ''} | {pct(r.mean_return)} | {pct(r.median_return)} | {r.up_frequency:.0%} | {pct(r.usual_return)} | {100*r.difference_from_usual:+.2f} | {pct(r.mean_market_excess)} |")
    lines += ["", "*Fewer than 10 events: small sample. Up frequency is historical, not a calibrated future probability. Usual mean uses all eligible dates, not a matched causal control group.", "",
              "## Historical conditions to monitor for upside", ""]
    supportive = events[(events.horizon == f["forecast_horizon"]) & (events.events >= 10)
                        & (events.mean_return > 0) & (events.difference_from_usual > 0)]
    if supportive.empty:
        lines.append("No jump/dive condition met the minimum of 10 events and a positive average response above the usual return at the forecast horizon.")
    else:
        lines.append("These are descriptive candidates found across many comparisons, not confirmed trading signals. Prioritized by event count.")
        for r in supportive.sort_values(["events", "difference_from_usual"], ascending=[False, False]).head(5).itertuples():
            lines.append(f"- After **{r.peer} {r.direction}s**, {ticker} averaged **{pct(r.mean_return)} over {r.horizon} days**, versus {pct(r.usual_return)} normally; {r.events} events, {r.up_frequency:.0%} positive. Shared market moves or chance may explain this pattern.")
    lines += ["",
              "## Do peer moves add forecasting information?", "",
              "Fixed Ridge penalty 10; same-date purged 60/20/20 chronological split within the recent window. Each peer is added separately to own returns and market/sector returns. Selection uses validation only.", "",
              "| Model | Validation RMSE | Test RMSE | Test squared-error improvement vs market/own |",
              "|---|---:|---:|---:|"]
    displayed_models = comparisons.sort_values("validation_rmse").head(15).configuration.tolist()
    displayed_models += ["mean", "market_and_own"]
    for r in comparisons[comparisons.configuration.isin(displayed_models)].sort_values("validation_rmse").itertuples():
        lines.append(f"| {r.configuration} | {r.validation_rmse:.4f} | {r.test_rmse:.4f} | {pct(r.test_mse_improvement_vs_market_and_own)} |")
    lines += ["", "## Factors that could support an upward move", "",
              "These are conditional responses of the latest refitted model, holding other inputs fixed. They do not predict that the inputs themselves will rise. A +1pp change means, for example, a daily return increasing from 1% to 2%.", ""]
    if not f["factors"]:
        lines.append("The historical mean won selection. There are no supported factor-based upside predictions from this fit.")
    for item in sorted(f["factors"], key=lambda x: x["forecast_change_for_plus_1pp"], reverse=True):
        change = item["forecast_change_for_plus_1pp"]
        direction = "increase" if change >= 0 else "decrease"
        lines.append(f"- **{item['factor']}**: a 1pp {direction} corresponds to **{abs(change)*100:.3f} percentage points higher** predicted {f['forecast_horizon']}-day return. Current contribution versus the training mean: {item['contribution']*100:+.3f}pp.")
    lines += ["", f"Latest fit audit: {f['latest_fit']}", f"Evaluation samples: {f['evaluation']}", "",
              "## Limits", "",
              "All results are exploratory. Same-day co-movement, event averages and predictive improvements do not establish causation. Multiple peer comparisons can produce chance patterns; the short test is not independent confirmation across market regimes.",
              "Events across different peers can share dates. Non-overlap within one peer does not make all events independent. SOXX may itself contain the target stock.",
              "SEC-only text mentions are excluded unless also industry/manual/benchmark candidates; current Yahoo membership still has survivorship bias. Review candidate evidence before treating a relationship as economic fact.",
              "The forecast uses after-close inputs. Its hypothetical scenarios are not executable trades or calibrated up probabilities."]
    lines += [f"- {w}" for w in report.get("warnings", [])]
    return "\n".join(lines)
