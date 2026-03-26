# StockAnalyzer
ML-driven stock analysis pipeline that combines market data and company news to predict short-term direction, run a strategy backtest, and generate an analyst-style report.

## What is implemented
- Price ingestion from `yfinance`.
- News ingestion from CSV (`published_at`, `title`, `content`, `source`, `ticker`).
- Trading-day alignment for news timestamps:
  - Weekend news rolls to next business day.
  - After 4:00pm ET news rolls to next business day.
- Richer news features:
  - Sentiment (`FinBERT` optional, lexicon fallback), sentiment dispersion, positive/negative counts.
  - Event counts (`earnings`, `product`, `legal`, `analyst`).
  - Volume and rolling features (`news_count_3d/7d`, `weighted_sentiment_3d/7d`).
- Time-based training split (no random split).
- Validation threshold tuning.
- Walk-forward evaluation by year.
- Backtest with transaction costs.
- Report diagnostics with non-predictive warning.

## Project structure
```text
StockAnalyzer/
  data/
    raw/
      news_template.csv
    processed/
  models/
  reports/
    curves/
    metrics/
    markdown/
  scripts/
    prepare_data.py
    train.py
    backtest.py
    report.py
    run_universe.py
  src/stock_analyzer/
    config.py
    data.py
    sentiment.py
    features.py
    modeling.py
    backtest.py
    reporting.py
```

## Setup
```powershell
cd StockAnalyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional FinBERT:
```powershell
pip install transformers torch
```

## Input news format
Use `data/raw/news_template.csv` as template. Minimum required columns:
- `published_at` (ISO timestamp)
- `title`
- `content`

Recommended columns:
- `source`
- `ticker`

Example row:
```csv
published_at,title,content,source,ticker
2025-01-28T21:10:00Z,Company raises guidance,Management raised full-year guidance after earnings,ExampleWire,NVDA
```

## Single ticker flow (NVDA example)
1. Prepare dataset + diagnostics:
```powershell
python scripts/prepare_data.py --ticker NVDA --start-date 2019-01-01 --end-date 2026-01-01 --news-csv data/raw/news_template.csv --out data/processed/dataset.csv --stats-out data/processed/dataset_stats.json
```

2. Train + tune threshold + walk-forward:
```powershell
python scripts/train.py --dataset data/processed/dataset.csv --train-end 2023-12-31 --valid-end 2024-12-31 --model-out models/baseline_model.joblib --metrics-out reports/metrics/metrics.json --walk-forward-start-year 2019 --walk-forward-end-year 2025
```

3. Backtest (uses tuned threshold from `reports/metrics/metrics.json` if `--prob-threshold` is omitted):
```powershell
python scripts/backtest.py --dataset data/processed/dataset.csv --model models/baseline_model.joblib --model-metrics reports/metrics/metrics.json --start-date 2025-01-01 --trade-cost-bps 5 --equity-out reports/curves/equity_curve.csv --metrics-out reports/metrics/backtest_metrics.json
```

4. Generate report:
```powershell
python scripts/report.py --ticker NVDA --equity-curve reports/curves/equity_curve.csv --model models/baseline_model.joblib --model-metrics reports/metrics/metrics.json --backtest-metrics reports/metrics/backtest_metrics.json --out reports/markdown/report.md
```

## Multiple companies in one run
```powershell
python scripts/run_universe.py --tickers NVDA AAPL MSFT TSLA --news-csv data/raw/news.csv --start-date 2019-01-01 --end-date 2026-01-01 --train-end 2023-12-31 --valid-end 2024-12-31 --backtest-start 2025-01-01 --summary-out reports/universe_summary.csv
```

Outputs per ticker:
- `data/processed/<ticker>_dataset.csv`
- `models/<ticker>_model.joblib`
- `reports/metrics/<ticker>_metrics.json`
- `reports/curves/<ticker>_equity_curve.csv`
- `reports/metrics/<ticker>_backtest_metrics.json`
- `reports/markdown/<ticker>_report.md`

## How to read each report
- `test_roc_auc`:
  - Around `0.50` means near-random signal.
  - `>= 0.55` is a reasonable early research threshold.
- `model_quality_flag`:
  - `non_predictive` means do not trust directional predictions yet.
  - `predictive` means model has some out-of-sample signal, still requires caution.
- Backtest:
  - Compare `strategy_total_return` vs `buy_hold_total_return`.
  - Check `max_drawdown` and `sharpe_estimate` before considering the strategy usable.
- Feature importances:
  - If news features are near zero, your news dataset is likely too sparse/noisy or poorly aligned.

## Common troubleshooting
- News features are all zero:
  - Ensure `ticker` values in news file exactly match symbols (e.g. `TSLA`).
  - Ensure timestamps are valid ISO strings and not empty.
- ROC-AUC stays near 0.50:
  - Increase quality/coverage of news, use richer sources, and try `--prefer-finbert`.
- Backtest looks good but ROC-AUC is weak:
  - Usually unstable luck. Re-check walk-forward metrics and multiple years.

## Important notes
- This is a research baseline, not financial advice.
- If `test_roc_auc` is near 0.50, treat signals as non-predictive.
- Improve news quality before trusting any strategy decisions.
