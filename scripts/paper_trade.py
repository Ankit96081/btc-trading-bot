"""Paper trade a strategy on live Binance Spot bars, with no real orders.

Example:
    python scripts/paper_trade.py --strategy ma_crossover \
        --param fast_period=24 --param slow_period=96 --stop-atr 3 --max-drawdown 0.25

Only Binance's public market-data endpoint is used; no API keys are needed.
State is kept in SQLite, so stopping and restarting with the same --run-id
resumes the run.  Create the --kill-file (default data/paper/KILL) to close
the position and halt the run permanently.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.cli import (
    add_cost_arguments,
    add_risk_arguments,
    add_strategy_arguments,
    cost_model_from,
    engine_config_from,
    parse_key_values,
    risk_config_from,
)
from src.core.logging_setup import configure_logging
from src.data.providers.binance_spot import BinanceSpotMarketDataProvider
from src.database.store import TradingStore
from src.execution.simulated import SimulatedBroker
from src.paper.feed import ClosedBarFeed
from src.paper.trader import LiveTrader, RunConflictError, TraderSettings
from src.strategies.registry import StrategyConfigurationError, strategy_factory

STATE_DIR = PROJECT_ROOT / "data" / "paper"
LOG_DIR = PROJECT_ROOT / "logs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_strategy_arguments(parser, "ma_crossover")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="Binance kline interval (default: 1h).")
    parser.add_argument("--run-id", help="Name of the run to create or resume.")
    parser.add_argument("--db", type=Path, default=STATE_DIR / "trading.db")
    parser.add_argument("--kill-file", type=Path, default=STATE_DIR / "KILL")
    parser.add_argument("--warmup-bars", type=int, default=500)
    parser.add_argument(
        "--once", action="store_true", help="Process the bars available now, then exit."
    )
    add_cost_arguments(parser)
    add_risk_arguments(parser)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    parameters = parse_key_values(args.param)
    run_id = args.run_id or f"paper-{args.strategy}-{args.timeframe}"
    configure_logging(LOG_DIR / f"{run_id}.log")

    try:
        strategy_factory(args.strategy, parameters)
        settings = TraderSettings(
            mode="paper",
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy=args.strategy,
            parameters=parameters,
            initial_cash=args.initial_cash,
            cost_model=cost_model_from(args),
            risk_config=risk_config_from(args),
            engine_config=engine_config_from(args),
            warmup_bars=args.warmup_bars,
            backfill_missed_bars=True,
        )
        store = TradingStore(args.db)
        feed = ClosedBarFeed(BinanceSpotMarketDataProvider(), args.symbol, args.timeframe)
        trader = LiveTrader(
            store=store,
            feed=feed,
            broker=SimulatedBroker(settings.cost_model),
            settings=settings,
            run_id=run_id,
            heartbeat_path=LOG_DIR / f"{run_id}.heartbeat.json",
            kill_file=args.kill_file,
        )
        trader.start()
    except (RunConflictError, StrategyConfigurationError, ValueError, OSError) as error:
        print(f"Paper trading failed to start: {error}", file=sys.stderr)
        return 1

    if args.once:
        processed = trader.step()
        print(f"Processed {processed} new bar(s) for run {run_id}.")
        return 0
    try:
        trader.run_forever()
    except KeyboardInterrupt:
        print("Stopped; restart with the same --run-id to resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
