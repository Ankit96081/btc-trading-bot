# Research Log

Every walk-forward run is recorded here, including failures. The go-live
checklist asks for the number of variants tried: the more runs, the more
likely a pass is luck. Runs are written down **before** they are executed,
and results are filled in afterwards without editing the plan.

Unless stated otherwise every run uses the defaults in
[validation_criteria.md](validation_criteria.md): 1h bars, 365-day rolling
train windows, 90-day test windows (12 folds, OOS 2023-01-01 to 2025-12-16),
0.1% fee, 2 bps slippage, holdout from 2026-03-01 untouched.

"Risk limits" = `--stop-atr 3 --max-drawdown 0.25 --max-daily-loss 0.05`.

## Runs 1–4 (before 2026-10-17; reproduced on 2026-10-17)

| # | Strategy | Grid | Sizing | Criteria | Verdict |
|---|---|---|---|---|---|
| 1 | ma_crossover | fast 12,24,48 × slow 96,168,336 | full | 4/9 | FAIL |
| 2 | ma_crossover + risk limits | same | vol target 0.4 | 5/9 | FAIL |
| 3 | donchian_breakout | entry 24,48,96,168 × exit 12,24,48 | full | 4/9 | FAIL |
| 4 | donchian_breakout + risk limits | same | vol target 0.4 | 7/9 | FAIL |

Run 4 failed on Sharpe vs buy-and-hold (0.98 vs 1.42) and profitable folds
(58% vs 60%).

## Runs 5–10 (planned 2026-10-17, before any results)

Aimed at the two criteria run 4 failed. New strategies are in
`src/strategies/trend.py`. `risk.vol_target` in a grid is tuned per fold on
the train window only. All runs use risk limits and `--warmup-bars 5000` so
200-day indicators are formed when each test window starts.

| # | Strategy | Grid | Sizing |
|---|---|---|---|
| 5 | trend_filter | sma_days 50,100,150,200 × band 0,0.02,0.05 | full (no vol target) |
| 6 | trend_filter | same | risk.vol_target 0.3,0.5,0.8 tuned |
| 7 | multi_timeframe_trend | ema_period 20,50,100 × min_votes 1,2,3 | vol target 0.4 |
| 8 | multi_timeframe_trend | same | risk.vol_target 0.3,0.5,0.8 tuned |
| 9 | donchian_breakout | entry 24,48,96,168 × exit 12,24,48 | risk.vol_target 0.3,0.5,0.8 tuned |
| 10 | donchian_breakout | same | risk.vol_target 0.3,0.5,0.8 × risk.vol_lookback 72,168,336 tuned |

Known limitation: train windows start without warm-up, so long-period
settings have fewer formed bars in-sample, which biases selection slightly
towards shorter periods.

### Results (2026-10-17)

| # | Return | Sharpe | Max DD | Profitable folds | Trips | 2× cost Sharpe | Robustness | Criteria | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 5 | +17% | 0.37 | 26.5% | 25% | 30 | 0.25 | 0.33 | 2/9 | FAIL |
| 6 | +13% | 0.31 | 26.5% | 25% | 29 | 0.19 | 0.37 | 1/9 | FAIL |
| 7 | +16% | 0.33 | 27.6% | 42% | 265 | −0.52 | 0.11 | 3/9 | FAIL |
| 8 | +58% | 0.72 | 29.9% | 42% | 243 | 0.13 | 0.36 | 3/9 | FAIL |
| 9 | +48% | 0.83 | 19.2% | 50% | 98 | 0.45 | 0.63 | 6/9 | FAIL |
| 10 | +37% | 0.69 | 19.7% | 50% | 95 | 0.31 | 0.63 | 5/9 | FAIL |

Buy-and-hold over the same OOS period: +419%, Sharpe 1.42, max DD 34.8%.

Observations:

- Runs 5–6 were in the market only ~21% of a strong bull market. Cause: the
  3× ATR stop is computed on **hourly** bars, so it sits ~1–2% below entry,
  far too tight for a daily trend filter. After a stop-out the risk manager
  waits for a flat signal before re-entering, which a 50–200-day filter may
  not give for months (9 of 16 exits in a diagnostic backtest were stops).
  This was a design error in the plan, not a code bug. Diagnosing it used a
  backtest over the OOS period, so the follow-up runs 11–12 are extra
  variants informed by OOS data and are counted as such.
- Runs 7–8: the 1h vote flips often; ~1 fill/day costs ~$6–7.6k in fees and
  the edge does not survive 2× costs.
- Runs 9–10: tuning volatility sizing per fold did not help Donchian; run 4
  (fixed 0.4) remains the best candidate (7/9).

## Runs 11–12 (planned 2026-10-17, after runs 5–10)

Same as runs 5–6 but **without `--stop-atr`** (drawdown and daily-loss limits
kept): `--max-drawdown 0.25 --max-daily-loss 0.05 --warmup-bars 5000`.

| # | Strategy | Grid | Sizing |
|---|---|---|---|
| 11 | trend_filter | sma_days 50,100,150,200 × band 0,0.02,0.05 | full |
| 12 | trend_filter | same | risk.vol_target 0.3,0.5,0.8 tuned |

### Results (2026-10-17)

| # | Return | Sharpe | Max DD | Profitable folds | Trips | 2× cost Sharpe | Robustness | Criteria | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11 | +160% | 1.06 | 40.4% | 50% | 50 | 0.97 | 0.44 | 5/9 | FAIL |
| 12 | +128% | 1.05 | 26.7% | 50% | 38 | 0.95 | 0.45 | 5/9 | FAIL |

Observations:

- Without the hourly stop the trend filter is in the market ~67% of the
  time and is the most cost-robust idea so far (2× cost Sharpe ≈ 0.96 vs
  0.61 for run 4). It still falls well short of buy-and-hold's Sharpe, and
  its drawdowns come from whipsaws around the average (folds 3, 6, 9).
- The chosen `sma_days` jumps between 50 and 200 from fold to fold and the
  neighbor ratio is < 0.5: the in-sample optimum is not stable.
- Volatility targeting in run 12 cut the drawdown (40% → 27%) at the cost
  of return, but not enough to pass.

## Status after 12 runs

No strategy passes. Best by criteria count: run 4 (7/9). Most cost-robust:
run 11. **Variants tried so far: 12.** The holdout has not been touched.
