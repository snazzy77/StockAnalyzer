# Running and understanding StockAnalyzer

This guide contains commands and methodology for the current analyzer. For the
project vision, see the [README](../README.md). Run commands from the repository root.

## Quick start

```powershell
python -m venv .venv-regression
.\.venv-regression\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-regression\Scripts\python.exe scripts/regression.py --ticker NVDA --mode reactions
.\.venv-regression\Scripts\python.exe scripts/regression.py --ticker NVDA --mode rolling --horizons 63
```

Replace `NVDA` with the Yahoo Finance ticker you want to research. Coverage and
history vary by company; insufficient data can prevent analysis. SEC discovery
is optional. The existing ignored `sec.local.json` contact configuration is read
automatically; without a contact, SEC discovery is skipped with a warning.

Use `--help` for all options. The dated offline examples below replay an existing
cached run; a fresh checkout must first download data without `--offline`.

## Related-stock reactions and conditional upside analysis

Use `--mode reactions` to answer: which related stocks move with the target,
what usually happens **after** their jumps/dives, and whether their moves improve
a forward-return forecast beyond the target's own history and market conditions.

```powershell
.\.venv-regression\Scripts\python.exe scripts/regression.py --ticker NVDA --mode reactions --start-date 2017-01-01 --end-date 2026-10-27 --train-end 2021-12-31 --valid-end 2023-12-31 --offline
```

This reuses the saved discovery snapshot and cached Yahoo prices. Omit `--offline`
to download/discover first. Analysis defaults to the **last four quarters**, a
**3% daily jump/dive**, responses over **1, 5 and 21 trading days**, and a **five-day
conditional forecast**. Customize with `--analysis-quarters 2`,
`--jump-threshold 0.04`, `--reaction-horizons 1 5 21`, or `--forecast-days 5`.
Long forecast horizons with a short analysis window may leave too few completed
evaluation observations and will produce an explicit error.

The report contains:

- NVDA's recent return versus comparison returns, same-day correlations and
  correlations after removing linear SPY/SOXX effects.
- For peer jumps and dives: event count, NVDA's subsequent mean/median return,
  historical positive-return frequency, usual return and market-excess return.
  The CSV also contains response quantiles, same-day returns and event dates.
- Historical upside conditions with at least ten events, positive mean response
  and improvement over the usual return. These are descriptive findings, not
  independently validated signals.
- A separate predictive comparison: mean baseline, own/market/sector model, and
  each peer added individually. Standardized Ridge uses fixed alpha 10. The recent
  window is split chronologically 60/20/20, with labels crossing boundaries purged.
  Peers require complete coverage on comparison dates. Validation chooses the
  configuration; test results only assess it. At least 50 training and 10 validation
  and test observations are required.
- A latest refit using completed recent outcomes, its expected return, current
  feature contributions, and hypothetical changes in the forecast if an input
  rises by one percentage point while others stay fixed. These scenarios do not
  predict that the input itself will rise. If the mean wins, no factor-based upside
  forecast is claimed.

Events start at the peer event day's **close** and end at a later target close.
For each peer/horizon the first qualifying event is kept, then overlapping windows
are skipped across both positive and negative events. Different peers can still
share events. Small samples are flagged; frequencies are not calibrated future
probabilities. Benchmark-adjusted co-movement is not causal influence, and SOXX
can contain the target stock itself. Many peer comparisons create selection risk;
short test-period improvements remain exploratory.

Only industry, benchmark and explicitly supplied candidates enter this analysis.
SEC-only text mentions are retained in the discovery evidence but excluded from
automatic peer inference. Add a reviewed company with `--candidates` during a
discovery run if appropriate. Company classifications are still current snapshots.

Outputs in `reports/regression/<ticker>/reactions/`: `reactions.md`, `reactions.json`,
`reactions_events.csv`, `reactions_associations.csv`, `reactions_model_comparison.csv`,
`reactions_predictions.csv`, and `reactions_model.joblib`. Markdown shows a limited
number of comparisons for readability; CSVs contain all evaluated candidates.
`--train-end` controls the saved discovery cutoff; it does not define this mode's
forecast training period, which is the recent window. `--valid-end` belongs to
the other modes; reactions uses its own chronological 60/20/20 split.

## Rolling recent-history regression (default)

`scripts/regression.py` defaults to **monthly walk-forward Ridge regression**.
It compares two-, three-, and four-quarter training windows and Ridge penalties
1, 10 and 100. A quarter means approximately 63 trading observations, not a
calendar quarter. Production selection is restricted to those recent windows
and a four-quarter mean-return fallback. A twelve-quarter Ridge model (alpha 10)
is also evaluated as a diagnostic benchmark, but cannot become the production model.

```powershell
.\.venv-regression\Scripts\python.exe scripts/regression.py --ticker NVDA --start-date 2017-01-01 --end-date 2026-10-27 --train-end 2021-12-31 --valid-end 2023-12-31 --offline
```

Omit `--offline` for initial discovery/downloads. Install dependencies using the
setup instructions below. Offline mode reuses the existing candidate snapshot
and exact date-range price cache. Optional SEC discovery still requires a real
contact email in `SEC_USER_AGENT`.

You can also save the contact in project-root `sec.local.json`:

```json
{"SEC_USER_AGENT": "StockAnalyzer your.name@example.com"}
```

This local file is ignored by Git and loaded automatically. An environment variable
overrides it. `--offline` and `--no-sec` still disable live SEC discovery.

### What recent training means

At each monthly refit, only feature-origin dates in the last `quarters * 63`
trading observations are eligible. Their complete future return must already
be known by that day's close. The code does **not** extend the window backwards
to replace immature labels. For a 63-day forecast, a two-quarter window therefore
contains approximately 63 usable origins; a four-quarter window contains 189.
Overlapping targets are not independent quarterly outcomes.

Historical features may use older prices, but older origins are not used to fit
the coefficients. Each fit uses up to five predefined core inputs (21-day
momentum, relative volume, market return, market volatility and optional VIX level)
and up to two screened peers, capped at six inputs total. Scaling, peer screening
and fitting are repeated using only that window.
At least 40 complete observations are required; skipped monthly fits are audited.

### Evaluation and the latest forecast

`--train-end` now marks the end of the initial warm-up period/start of validation,
not a permanent training cutoff. `--valid-end` ends configuration selection.
Every model refits at the first observed trading date of each month, using
information available after that close. Historical test models may learn from
earlier test-period outcomes **only after those outcomes have completed**, which
simulates ongoing deployment. Validation targets crossing `valid-end` are excluded.
Configurations are compared on the same complete forecast dates.

Validation RMSE chooses a fixed recent-window configuration or the rolling mean
fallback; test outcomes never choose the window or penalty. The latest model is
then **refitted separately as of the final input date**, using recent completed
outcomes, and saved in the model artifact. Reports expose its exact origin dates,
last completed target date, row count and non-overlapping-origin count.

The report marks `no_useful_signal` if the baseline wins or the chosen model fails
to beat the rolling baseline on test data. A forecast may still be shown for
research; that flag does not claim it is actionable. An empirical 80% error range
uses non-overlapping validation residuals when at least eight are available. It is
not a guaranteed prediction interval and may be unreliable after a regime change.

Ridge coefficients and per-forecast contributions are reported. **Ordinary OLS
p-values, t-tests and F-tests do not apply to these Ridge coefficients.** For the
older exploratory HAC/OLS diagnostics use `--mode static` below.

Options: `--training-quarters 2 3 4`, `--ridge-alphas 1 10 100`,
`--benchmark-quarters 12`, `--max-peers 2` (rolling mode caps this at two).
Outputs in `reports/regression/<ticker>/rolling/` use the prefix `rolling_21d` / `rolling_63d` and include Markdown, JSON,
coefficient/prediction CSVs and a latest-refit model. JSON includes every monthly
fit's audit. Static artifacts are preserved. Candidate discovery remains fixed
for a run; rolling training does not remove the documented Yahoo universe bias
or automatically refresh SEC relationships every month.

## Static return regression and automatic comparisons

The `scripts/regression.py --mode static` workflow predicts **21- and 63-trading-day returns**
(roughly one month and one quarter). It is an optional diagnostic mode within the current analyzer.

### Install and run

```powershell
.\.venv-regression\Scripts\python.exe -m pip install -r requirements.txt
# Optional, required only for SEC discovery. Use your own real contact address.
$env:SEC_USER_AGENT = "StockAnalyzer your.name@example.com"
.\.venv-regression\Scripts\python.exe scripts/regression.py --mode static --ticker NVDA --start-date 2017-01-01 --end-date 2026-10-27 --train-end 2021-12-31 --valid-end 2023-12-31
```

The end date is exclusive. This example provides five years of raw training
history, a validation period, and a later test period. Rolling features consume
the first 252 observations. Choose boundaries with enough history for every split.
Features are available after the daily close; this is not an intraday signal.

No paid API or SEC API key is required. The official SEC EDGAR service is distinct
from the commercial `sec-api.io` service, which this project does not use.
SEC requests identify the app/contact, are paced at four requests/second or less,
retry transient errors, and cache responses for 24 hours.

### How comparisons are discovered

- A fixed benchmark panel covers the broad market, growth stocks, semiconductors,
  and all eleven major equity sectors, including energy, utilities and industrials.
  SPY is required; VIX is optional and missing downloads are reported.
- Yahoo industry metadata supplies candidate peers automatically when supported
  by the installed `yfinance` version and available from Yahoo.
- SEC discovery scans up to four latest 10-K/10-Q/20-F filings **published before
  the training cutoff**, following historical submission pages when necessary.
  It matches SEC company names near competitor, supplier, customer or partner
  language. Each match retains its passage, filing date and URL.
- SEC matches are **unverified relationship candidates**. The context does not
  establish relationship direction, importance, or causality. Name matching can
  miss aliases or produce false positives. This initial implementation scans the
  target company's filings, not reverse disclosures across every other company.
- Candidate returns are measured relative to SPY over 21 trading days. Their
  correlation with future target returns must have the same sign in both halves
  of training. At most four peer/sector features survive the default selection;
  features correlated above 0.90 in absolute value with an already retained input
  are excluded. This is a screening heuristic, not a causal relationship test or
  a full market-beta residualization model.

Yahoo classifications and SEC ticker/name mappings are current snapshots.
Historical filing cutoffs prevent future filing evidence entering selection, but
do **not** eliminate universe/survivorship bias. Candidate snapshots include a
retrieval timestamp and are saved for audit and offline replay.

### Model and evaluation

Ordinary least squares (statsmodels) uses adjusted price returns over 5, 21, 63
and 252 trading days, 21-day volatility, five-day volume change, relative volume,
market returns/volatility, optional VIX level/change, and selected comparison inputs.
Predictors are standardized using training statistics only. Missing data is never
backfilled; sparse/constant/redundant training inputs are excluded and remaining
incomplete rows are dropped. Downloads use adjusted prices consistently in this
workflow; cached data is reused by ticker and date range.

Three variants are compared: **own_only**, **market**, and **adjacency**. Validation
RMSE selects the variant; the test set is not used for selection. Training targets
must finish by `train-end`, and validation targets by `valid-end`, purging overlapping
labels across boundaries. Monthly and quarterly horizons are evaluated separately;
their errors are on different return scales and should not be compared directly.

Reports include coefficients, HAC standard errors, t-values, p-values, 95% coefficient
intervals, joint robust F-statistic/p-value, training RÂ² and adjusted RÂ². HAC uses
`horizon - 1` lags to account for overlapping return windows. These are exploratory
statistics after feature selection, not multiple-testing-adjusted significance or
proof of causality. Check coefficient stability before interpreting individual inputs.

Validation/test outputs include RMSE, MAE, direction accuracy and out-of-sample RÂ²
relative to the **training mean return** baseline (negative means worse than that
baseline). Test reports also include a non-overlapping-origin RMSE and sample count.
The model is fitted on training data only; no rolling refit or automatic production
retraining is performed. The latest forecast uses the last input row, including rows
without future labels; if required inputs are missing it is reported as unavailable.
Its feature contributions sum to the predicted return and describe model arithmetic.

These forecasts are expected returns, **not probabilities of an up move**. This
workflow does not claim trading profitability or produce a transaction-cost backtest.
The retired v1 classifier and its backtest are no longer part of this repository.

### Options and outputs

```powershell
# Add explicit candidates, or disable either discovery source.
.\.venv-regression\Scripts\python.exe scripts/regression.py --mode static --ticker NVDA --candidates AMD TSM META ORCL VRT ETN --no-yahoo-discovery --no-sec
# Replay with exactly the same dates and the saved candidate snapshot/cache.
.\.venv-regression\Scripts\python.exe scripts/regression.py --mode static --ticker NVDA --start-date 2017-01-01 --end-date 2026-10-27 --train-end 2021-12-31 --valid-end 2023-12-31 --offline
# Regression and evidence checks (no network required).
.\.venv-regression\Scripts\python.exe -m unittest discover -s tests -v
```

Other options: `--horizons 21 63`, `--max-peers 4`, `--cache-dir`, `--out-dir`.
If SEC contact information is absent, discovery is skipped with a visible warning;
Yahoo metadata/optional ticker failures also appear in the report. Missing target
or SPY history, insufficient observations, and rank-deficient fits stop the run.
Offline replay requires matching dates, saved candidates and price cache; it never
makes network requests. An explicit `--end-date` is recommended for reproducibility.

Outputs under `reports/regression/<ticker>/`:

- `candidates.json`: source labels, timestamps, SEC evidence and discovery warnings.
- `static/21d.md` / `static/63d.md`: readable model comparisons and explanations.
- `static/21d.json` / `static/63d.json`: diagnostics, selected inputs and prediction contributions.
- `static/*_coefficients.csv`, `static/*_predictions.csv`, `static/*_model.joblib`: fitted statistics,
  held-out forecasts, and the model with its training scaling parameters.

Data is cached under `data/cache/regression/`. Use a new output directory to preserve
an older research run; rerunning into the same directory replaces its artifacts.

References: [SEC API access](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[Yahoo industry interface](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Industry.html),
[statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).


Each ticker folder groups artifacts into `reactions/`, `rolling/`, and `static/`.
The shared `candidates.json` stays at the ticker root for all modes and offline replay.
