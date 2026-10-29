# StockAnalyzer

Understand a stock through the companies and markets around it.

StockAnalyzer is a research project exploring how a company's price behavior
relates to its industry, the broader market, and other potentially connected
businesses. Its vision is to turn those relationships into clear, testable
explanations: what has happened before, which inputs improve a forecast, and
when the evidence is too weak to draw a useful conclusion.

## The questions we want to answer

- Which companies and sectors are relevant to a stock I choose?
- When a related stock jumps or falls, how does this stock usually respond over
  the following days or weeks?
- Does that relationship add predictive information beyond the stock's own
  history and overall market conditions?
- Which observed conditions are associated with stronger future returns, and
  how consistent is that evidence?
- Does a model provide useful information beyond a simple historical average?

NVDA has been the initial research case. The pipeline accepts other tickers;
support depends on available price history and comparison data. The broader goal
is a useful research workflow for any chosen stock, with relationships supported
by evidence rather than a manually prescribed list of peers.

## What the project does today

**Discover comparisons.** Yahoo Finance supplies adjusted prices and industry
candidates. A benchmark panel covers the market and major sectors, including
technology, semiconductors, energy, and utilities. Optional SEC EDGAR searches
retain filing passages as evidence for potential business relationships.
SEC text matches are unverified; SEC-only candidates are excluded from automatic
reaction analysis unless explicitly supplied for comparison.

**Explain historical reactions.** The analyzer measures co-movement, including
correlations adjusted for market and semiconductor effects. For related-stock
jumps and dives, it reports subsequent average and median returns, positive-return
frequencies, event counts, and comparisons with usual and market returns.

**Test predictive value.** Reaction analysis compares a historical-average
baseline, an own-stock/market/sector model, and models adding individual peers.
Its default forecast covers five trading days. A separate rolling Ridge workflow
produces monthly and quarterly return estimates using recent two- to four-quarter
training windows. Optional static OLS analysis provides exploratory coefficient,
p-value, t-statistic, F-statistic, and R-squared diagnostics.

**Expose the evidence.** Reports include model comparisons, evaluation errors,
training dates, feature contributions, and data warnings. A baseline can win;
the analyzer should say when a more elaborate model has not earned confidence.

## How to read the results

A historical relationship is an observation, not proof that one company moves
another. Positive-return frequencies describe past events; they are not calibrated
probabilities for the next trade. Forecast contributions explain the fitted model,
not verified causes of price changes.

Models are evaluated chronologically, with unfinished targets excluded across
split boundaries. Even so, short recent windows, overlapping returns, many peer
comparisons, and changing market conditions limit the evidence. Current company
classifications also introduce bias into historical comparisons.

The project remains an exploratory research tool. It has not established a
reliable trading advantage, and its return forecasts are not a portfolio backtest.
In particular, a quarterly forecast has far fewer independent outcomes available
for evaluation than the daily row count suggests.

## Where the project is heading

- Better relationship discovery, with reviewed evidence for competitors,
  customers, suppliers, and other business connections.
- Tests of delayed reactions and comparisons against similar market conditions.
- More reliable uncertainty estimates and evaluation across multiple market periods.
- Broader validation across companies, including models trained across stocks.
- Carefully evaluated nonlinear models, such as boosted trees or small neural
  networks, when they improve on the same baseline tests.

These are research directions, not implemented capabilities. Cloud training can
support larger workloads later; it is not required to analyze another ticker.

## Explore the project

The repository contains the current relationship and regression analyzer. The
retired v1 news-classification and strategy-backtest pipeline has been removed.

```text
scripts/regression.py          Analysis entry point
src/stock_analyzer/
  relationships.py             Comparison discovery and SEC evidence
  reactions.py                 Event reactions and conditional forecasts
  rolling.py                   Recent-window, walk-forward Ridge models
  regression.py                Shared features and static OLS diagnostics
tests/                         Model and data-boundary checks
docs/usage.md                  Commands and detailed methodology
data/cache/regression/         Local downloaded data (generated, ignored)
reports/regression/<ticker>/   Reports and model artifacts (generated, ignored)
```

Start with `reactions/reactions.md` for related-stock behavior, `rolling/rolling_21d.md` for monthly
forecasts, or `rolling/rolling_63d.md` for quarterly forecasts in the ticker's report folder.
These files appear after running the corresponding mode; they are not bundled
with a fresh checkout. CSV and JSON files contain the underlying detail.

For setup, commands, SEC contact configuration, and model methodology, see the
[usage guide](docs/usage.md).
