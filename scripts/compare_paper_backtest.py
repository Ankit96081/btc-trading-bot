"""Compare a paper or testnet run with a backtest over the bars it processed.

Example:
    python scripts/compare_paper_backtest.py --run-id paper-ma_crossover-1h

A paper run that never missed a bar should match the backtest exactly.
Differences point to restarts that skipped bars, or, on the testnet, to real
fill prices and fees differing from the cost model.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.runner import run_backtest
from src.database.store import TradingStore
from src.paper.trader import settings_from_record
from src.strategies.registry import strategy_factory


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "paper" / "trading.db")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(arguments)

    store = TradingStore(args.db)
    record = store.get_run(args.run_id)
    if record is None:
        print(f"No run named {args.run_id!r} in {args.db}", file=sys.stderr)
        return 1
    stored = store.bars(args.run_id)
    warmup = next((index for index, (_, traded) in enumerate(stored) if traded), len(stored))
    if warmup == len(stored):
        print("The run has not traded any bars yet.")
        return 0

    settings = settings_from_record(record)
    bars = [bar for bar, _ in stored]
    backtest = run_backtest(
        bars,
        strategy_factory(settings.strategy, settings.parameters),
        initial_cash=settings.initial_cash,
        cost_model=settings.cost_model,
        risk_config=settings.risk_config,
        engine_config=settings.engine_config,
        warmup_bars=warmup,
    )
    live_fills = store.fills(args.run_id)
    live_curve = store.equity_curve(args.run_id)

    print(f"Run {args.run_id}: {len(bars) - warmup} traded bars "
          f"({bars[warmup].open_time:%Y-%m-%d %H:%M} to {bars[-1].open_time:%Y-%m-%d %H:%M} UTC)")
    print(f"  {'':<16}{'Live':>16}{'Backtest':>16}")
    print(f"  {'Fills':<16}{len(live_fills):>16}{len(backtest.fills):>16}")
    live_equity = live_curve[-1].equity if live_curve else record.initial_cash
    print(f"  {'Final equity':<16}{live_equity:>16.2f}{backtest.equity_curve[-1].equity:>16.2f}")

    for index, (live, simulated) in enumerate(zip(live_fills, backtest.fills, strict=False)):
        if live != simulated:
            print(f"First differing fill (#{index + 1}):\n  live:     {live}\n  backtest: {simulated}")
            return 0
    if len(live_fills) != len(backtest.fills):
        print("Fills match as far as both go, but their counts differ.")
    else:
        print("Every fill matches the backtest exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
