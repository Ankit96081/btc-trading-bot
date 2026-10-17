# Strategy Validation Criteria

**Status:** fixed on 2026-09-26, *before* any walk-forward results were produced.
Changing a threshold after seeing results defeats its purpose. If a change is
ever needed, record the date and reason below, and re-run every candidate
against the new version.

These criteria decide whether a strategy may move from research to paper
trading. They are encoded in `src/validation/criteria.py` (`ValidationCriteria`),
and `scripts/walk_forward.py` checks them automatically.

## What is measured

All numbers come from **out-of-sample (OOS)** walk-forward results, **after fees
and slippage**:

- The data before the holdout period is split into rolling train/test windows.
- In each fold, parameters are chosen using only the train window. The chosen
  parameters are then run on the following, unseen test window.
- The test windows are joined into one continuous OOS equity curve. Each fold
  starts with the previous fold's ending equity.
- The benchmark is buy-and-hold BTC over exactly the same OOS period.

Default setup: 1h bars, 365-day train windows, 90-day test windows, a 0.1% fee
per fill, 2 bps slippage, $10,000 starting capital.

## Criteria (all must pass)

| # | Criterion | Threshold | Why |
|---|-----------|-----------|-----|
| 1 | OOS total return | > 0 | It must make money after costs. |
| 2 | OOS Sharpe ratio | ≥ 0.8 | The return must be worth its volatility. |
| 3 | OOS Sharpe vs buy-and-hold | ≥ buy-and-hold Sharpe | Otherwise just hold BTC. |
| 4 | OOS max drawdown | ≤ 25% | The drawdown must be survivable, psychologically and financially. |
| 5 | Profitable folds | ≥ 60% of folds | The edge must hold across market regimes, not come from one lucky period. |
| 6 | OOS round trips | ≥ 30 | Enough trades that the result is not just noise. |
| 7 | Fills per day | ≤ 10 | Frequency is a cap, not a target (see below). |
| 8 | Sharpe with 2× costs | ≥ 0.5 | The edge must survive worse fees or slippage than assumed. |
| 9 | Parameter robustness | median neighbor ratio ≥ 0.5 | Neighboring parameter sets must keep at least half the chosen set's train score. Otherwise the choice was a lucky spike. |

## Holdout

Data from **2026-03-01** onwards is the **final holdout**. Walk-forward tuning
and strategy development never use it. A strategy that passes all nine criteria
is evaluated on the holdout **once** (`scripts/walk_forward.py --evaluate-holdout`),
using the parameters chosen in the last fold. To pass, it needs a positive
return and a max drawdown of at most 25% on the holdout. If it fails, the
strategy is rejected. It is not tuned further on this data.

## Trade frequency and costs

The README targets roughly 5–10 trades a day. With a 0.1% taker fee plus
slippage, every full-size fill costs about 0.12% of equity:

| Full-size fills/day | Daily cost | Yearly cost (compounded) |
|---|---|---|
| 2 | 0.24% | −58% |
| 5 | 0.60% | −89% |
| 10 | 1.20% | −99% |

A strategy would need a very large per-trade edge to overcome this. So the
criteria cap frequency (criterion 7) instead of requiring it. Ways to trade
more often without paying this cost include smaller position sizes, maker
(limit) orders, and BNB fee discounts. Those are future research topics, not
assumptions made here.

## Change log

- 2026-09-26: initial version.
