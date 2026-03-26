from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def run_threshold_strategy(
    df: pd.DataFrame, model, prob_threshold: float = 0.55, trade_cost_bps: float = 5.0
) -> BacktestResult:
    dfx = df.copy().sort_values("date").reset_index(drop=True)

    probs = model.predict_proba(dfx[FEATURE_COLUMNS])[:, 1]
    dfx["prob_up"] = probs
    dfx["signal"] = (dfx["prob_up"] >= prob_threshold).astype(int)

    # Long-only signal. Strategy return uses forward return from feature pipeline.
    dfx["gross_strategy_ret"] = dfx["signal"] * dfx["forward_return"]

    turns = dfx["signal"].diff().abs().fillna(0)
    trade_cost = (trade_cost_bps / 10000.0) * turns
    dfx["net_strategy_ret"] = dfx["gross_strategy_ret"] - trade_cost

    dfx["equity"] = (1.0 + dfx["net_strategy_ret"]).cumprod()
    dfx["buy_hold_equity"] = (1.0 + dfx["forward_return"]).cumprod()

    mean_ret = float(dfx["net_strategy_ret"].mean())
    vol = float(dfx["net_strategy_ret"].std(ddof=0))
    sharpe = (mean_ret / vol) * np.sqrt(252 / 5) if vol > 0 else 0.0

    peak = dfx["equity"].cummax()
    drawdown = dfx["equity"] / peak - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    metrics = {
        "strategy_total_return": float(dfx["equity"].iloc[-1] - 1.0) if not dfx.empty else 0.0,
        "buy_hold_total_return": float(dfx["buy_hold_equity"].iloc[-1] - 1.0) if not dfx.empty else 0.0,
        "sharpe_estimate": float(sharpe),
        "max_drawdown": max_drawdown,
        "avg_signal_rate": float(dfx["signal"].mean()) if not dfx.empty else 0.0,
    }
    return BacktestResult(equity_curve=dfx, metrics=metrics)

