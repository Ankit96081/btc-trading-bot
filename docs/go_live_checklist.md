# Go-Live Checklist

No real funds may be used until **every** item below is checked and the
evidence is linked. The testnet client (`src/execution/binance_testnet.py`)
refuses any host other than `testnet.binance.vision`. Trading real money
requires a deliberate code change, and that change should only be made after
this checklist is complete.

Each checked box needs a date, the person who checked it, and a link to the
evidence (a report file, a log, a commit).

## 1. Strategy validation

- [ ] The strategy passes all nine criteria in
      [validation_criteria.md](validation_criteria.md), using
      `scripts/walk_forward.py`. Save the JSON report (`--report`).
- [ ] The criteria document was not changed after results were seen, or every
      change is logged with a date and reason and all candidates were re-run.
- [ ] The holdout evaluation (`--evaluate-holdout`) was run **once** and passed.
- [ ] The number of strategy variants tried is recorded. Many attempts inflate
      the chance that a pass is luck.

## 2. Paper trading

- [ ] At least 30 days of continuous paper trading (`scripts/paper_trade.py`)
      with the exact parameters that passed validation.
- [ ] `scripts/compare_paper_backtest.py` shows every fill matching the
      backtest, or each difference is explained (for example a restart).
- [ ] Heartbeat monitoring (`scripts/check_health.py`) ran on a schedule for
      the whole period and alerted when the trader stopped.
- [ ] The kill switch was tested: creating the kill file closed the position,
      and the halt survived a restart.

## 3. Binance Spot Testnet

- [ ] At least 4 weeks on the testnet (`scripts/testnet_trade.py`) with the
      same parameters and risk limits.
- [ ] Testnet fills compared with paper trading over the same weeks: average
      slippage and fees are within the cost model's assumptions (0.1% fee,
      2 bps slippage). Otherwise, update the cost model and re-validate.
- [ ] Restarts tested, including a kill -9 while an order was in flight. The
      bot recovered the order and reconciled its position without duplicates.
- [ ] No unexplained reconciliation warnings in the logs or `events` table.

## 4. Operations and safety

- [ ] Production API keys have **trading only**: withdrawals disabled, IP
      allow-list set. Keys come from environment variables and never from
      files in the repository.
- [ ] The first live allocation is small (for example ≤ 5% of the intended
      size), and scaling up is a separate, reviewed decision.
- [ ] Risk limits are set: `--max-drawdown`, `--max-daily-loss`, and a stop
      (`--stop-atr` or `--stop-pct`).
- [ ] Alerting reaches a human within minutes (heartbeat stale, error, halted).
- [ ] A written runbook covers: how to stop, how to flatten manually on the
      exchange, and how to restart.
- [ ] The production host keeps its clock synchronized (NTP).

## 5. Sign-off

- [ ] Reviewed by someone other than the strategy's author.
- [ ] Go-live date and initial size recorded here.
