# BTC Trading Bot

An automated BTC/USDT trading research and paper-trading system.

## Current Phase

Phase 5 - Strategy research and validation

The whole pipeline is built and tested: data, backtesting, risk management,
walk-forward validation, paper trading, and Binance Testnet execution. **No
strategy has passed the validation criteria yet** (see
[Research status](#research-status)). Until one does, paper and testnet
trading are for testing the software only.

## Objectives

- Historical BTC market data ✅
- Strategy research 🔄 (baselines and trend filters; none validated yet)
- Backtesting ✅
- Walk-forward validation ✅
- Paper trading ✅ (software built; waiting for a validated strategy)
- Binance Testnet integration ✅ (software built; waiting for a validated strategy)
- Risk management ✅
- Live trading only after extensive validation ⛔ gated by [docs/go_live_checklist.md](docs/go_live_checklist.md)

## Initial Trading Goals

- Low-frequency trading
- Approximately 5-10 trades/day as a target (see note below)
- Focus on risk-adjusted returns
- Minimize unnecessary trading
- Account for fees and slippage
- Avoid overfitting

**Note on trade frequency:** Each market order costs about 0.12% of equity
(0.1% fee plus slippage). At 5–10 full-size trades a day, that adds up to an
89–99% loss per year before any edge. The validation criteria therefore treat
frequency as a cap (≤ 10 fills/day), not a target. See
[docs/validation_criteria.md](docs/validation_criteria.md).

## Important

This project is initially a research and paper-trading system.

No real funds should be used until the strategy and software pass predefined validation criteria.

## Workflow

```bash
pip install pytest ruff   # dev tools; the project has no runtime dependencies

# 1. Download 1m history (resumable, retries transient errors, skips Binance's
#    known 2023-03-24 outage; one file per month works well)
python scripts/download_btc_history.py --start 2024-01-01T00:00:00Z \
    --end 2024-02-01T00:00:00Z --output data/raw/btcusdt_1m_2024-01.csv

# 2. Combine raw files, report exchange gaps, and resample to research timeframes
python scripts/build_dataset.py --timeframes 15m 1h 4h

# 3. Backtest (decides on bar close, fills at next bar open, charges fees + slippage)
python scripts/run_backtest.py --strategy donchian_breakout \
    --param entry_period=48 --param exit_period=24 --vol-target 0.4 --stop-atr 3

# 4. Walk-forward validate against the predefined criteria (holdout from 2026-03-01 untouched)
python scripts/walk_forward.py --strategy donchian_breakout \
    --grid entry_period=24,48,96,168 --grid exit_period=12,24,48 \
    --vol-target 0.4 --stop-atr 3 --max-drawdown 0.25 --max-daily-loss 0.05
#    Risk settings can be tuned per fold too: --grid risk.vol_target=0.3,0.5,0.8

# 5. Paper trade on live public Binance data (no keys; resumable; touch data/paper/KILL to halt)
python scripts/paper_trade.py --strategy donchian_breakout --param entry_period=48 ...
python scripts/compare_paper_backtest.py --run-id paper-donchian_breakout-1h
python scripts/check_health.py logs/paper-donchian_breakout-1h.heartbeat.json

# 6. Binance Spot Testnet (test funds; keys from BINANCE_TESTNET_API_KEY / _SECRET)
python scripts/testnet_trade.py --strategy donchian_breakout --initial-cash 1000 ...

pytest
ruff check . --no-cache
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/data/` | Candle models, validation, Binance public data provider, resumable downloader, CSV format, dataset loader, resampler |
| `src/strategies/` | `Strategy` protocol (target exposure from closed bars), indicators, baseline and trend strategies, registry |
| `src/risk/` | `RiskManager`: exposure cap, volatility targeting, stops, daily loss limit, drawdown kill switch, trade cap |
| `src/execution/` | Orders/fills, cost model, `Broker` protocol, `SimulatedBroker`, testnet client and broker |
| `src/core/` | `TradingEngine` (the one bar-by-bar loop every mode uses), `Portfolio`, heartbeat, logging |
| `src/backtest/` | Backtest runner, performance metrics, shared CLI helpers |
| `src/validation/` | Walk-forward optimization and the predefined pass/fail criteria |
| `src/paper/` | Closed-bar live feed, `LiveTrader` (paper and testnet), exchange reconciliation |
| `src/database/` | SQLite storage for live runs (bars, fills, equity, state, events) |
| `docs/` | Validation criteria, research log, and the go-live checklist |

## Research status

Data: BTC/USDT 1m, 2022-01 to 2026-09. The only gap is Binance's 2023-03-24
12:40–14:00 UTC outage; bars touching it are dropped, not filled in.

Walk-forward on 1h bars, with out-of-sample windows from 2023-01 to 2025-12
(12 folds of 90 days, 365-day training windows):

| Strategy | OOS return | Sharpe | Max DD | Criteria passed | Verdict |
|---|---|---|---|---|---|
| Buy & hold (benchmark) | +419% | 1.42 | 34.8% | — | — |
| MA crossover | +73% | 0.71 | 41.8% | 4/9 | FAIL |
| MA crossover + risk limits | +19% | 0.38 | 27.9% | 5/9 | FAIL |
| Donchian breakout | +30% | 0.45 | 35.6% | 4/9 | FAIL |
| Donchian breakout + risk limits | +65% | 0.98 | 18.4% | 7/9 | FAIL |
| Trend filter (daily SMA), no ATR stop | +160% | 1.06 | 40.4% | 5/9 | FAIL |
| Trend filter, tuned vol target, no ATR stop | +128% | 1.05 | 26.7% | 5/9 | FAIL |

"Risk limits" means `--vol-target 0.4 --stop-atr 3 --max-drawdown 0.25 --max-daily-loss 0.05`.
The best candidate roughly halves buy-and-hold's drawdown. It still falls short
of buy-and-hold's risk-adjusted return, and it has 58% profitable folds against
the 60% required. The criteria are fixed in advance and will not be loosened
to make a strategy pass.

Every run, including the multi-timeframe and volatility-sizing experiments
that did worse, is recorded in [docs/research_log.md](docs/research_log.md)
(12 variants so far).
