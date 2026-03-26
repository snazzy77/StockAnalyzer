from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_executive_summary(
    model_metrics: dict[str, float],
    backtest_metrics: dict[str, float],
    latest_row: pd.Series,
    diagnostics: dict[str, str | float] | None,
) -> list[str]:
    test_auc = _safe_float(model_metrics.get("test_roc_auc", 0.0))
    test_acc = _safe_float(model_metrics.get("test_accuracy", 0.0))
    signal_prob = _safe_float(latest_row.get("prob_up", 0.0))
    signal_on = int(_safe_float(latest_row.get("signal", 0.0)))
    strat_ret = _safe_float(backtest_metrics.get("strategy_total_return", 0.0))
    hold_ret = _safe_float(backtest_metrics.get("buy_hold_total_return", 0.0))
    sharpe = _safe_float(backtest_metrics.get("sharpe_estimate", 0.0))
    mdd = _safe_float(backtest_metrics.get("max_drawdown", 0.0))
    quality_flag = str((diagnostics or {}).get("model_quality_flag", "unknown")).lower()

    if quality_flag == "non_predictive" or test_auc < 0.53:
        reliability = "Model reliability is low (near-random out-of-sample discrimination)."
    elif test_auc < 0.58:
        reliability = "Model reliability is moderate; signal exists but is weak."
    else:
        reliability = "Model reliability is stronger than baseline and may contain useful signal."

    if strat_ret > hold_ret:
        perf = "Strategy outperformed buy-and-hold in the evaluated backtest period."
    else:
        perf = "Strategy underperformed buy-and-hold in the evaluated backtest period."

    if mdd <= -0.5:
        risk = "Risk is high due to deep drawdown behavior."
    elif sharpe < 0.5:
        risk = "Risk-adjusted performance is weak."
    else:
        risk = "Risk profile is comparatively more controlled in this backtest."

    if signal_on == 1 and signal_prob >= 0.6:
        signal_text = "Latest model output is a long signal with relatively high conviction."
    elif signal_on == 1:
        signal_text = "Latest model output is a long signal with modest conviction."
    else:
        signal_text = "Latest model output is neutral/no-position under the current threshold."

    caution = (
        "Treat this as research output, not trading advice. Validate on broader data before relying on it."
        if quality_flag == "non_predictive" or test_acc < 0.52
        else "Continue monitoring stability across future walk-forward windows."
    )

    return [
        "## Executive Summary",
        f"- {reliability}",
        f"- {perf}",
        f"- {risk}",
        f"- {signal_text}",
        f"- {caution}",
    ]


def write_markdown_report(
    path: Path,
    ticker: str,
    model_metrics: dict[str, float],
    backtest_metrics: dict[str, float],
    latest_row: pd.Series,
    top_features: list[tuple[str, float]],
    diagnostics: dict[str, str | float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# StockAnalyzer Report: {ticker}",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    lines.extend(_build_executive_summary(model_metrics, backtest_metrics, latest_row, diagnostics))
    lines.extend(
        [
            "",
        "## Latest Signal",
        f"- Date: {str(latest_row.get('date'))}",
        f"- Predicted probability of up move: {float(latest_row.get('prob_up', 0.0)):.3f}",
        f"- Strategy signal (1=long): {int(latest_row.get('signal', 0))}",
        "",
        "## Classification Metrics",
        ]
    )
    for k, v in model_metrics.items():
        lines.append(f"- {k}: {v:.4f}")

    lines.extend(["", "## Backtest Metrics"])
    for k, v in backtest_metrics.items():
        lines.append(f"- {k}: {v:.4f}")

    if diagnostics:
        lines.extend(["", "## Diagnostics"])
        for k, v in diagnostics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")

    lines.extend(["", "## Top Feature Importances"])
    for name, val in top_features:
        lines.append(f"- {name}: {val:.4f}")

    lines.extend(
        [
            "",
            "## Watchlist Indicators",
            "- Rising `news_count` with positive `weighted_sentiment` can support upside continuation.",
            "- Spikes in `legal_events` often increase downside risk and volatility.",
            "- Divergence between `ma_5_20_ratio` and sentiment can indicate fragile momentum.",
        ]
    )
    if diagnostics and str(diagnostics.get("model_quality_flag", "")).lower() == "non_predictive":
        lines.extend(
            [
                "",
                "## Warning",
                "- Current model appears non-predictive on out-of-sample data (near-random ROC-AUC).",
                "- Treat latest signal as low-confidence until data/features are improved.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def infer_feature_importance(model, feature_names: list[str]) -> list[tuple[str, float]]:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = list(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        importances = [abs(float(c)) for c in coef]

    if importances is None:
        return [(f, 0.0) for f in feature_names[:8]]

    pairs = list(zip(feature_names, map(float, importances)))
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:8]
