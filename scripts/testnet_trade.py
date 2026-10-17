"""Trade a strategy on the Binance Spot *Testnet* (test funds only).

Setup:
    1. Create API keys at https://testnet.binance.vision (log in with GitHub).
    2. Export them; they are never read from files or the command line:
         BINANCE_TESTNET_API_KEY=...   BINANCE_TESTNET_API_SECRET=...

Example:
    python scripts/testnet_trade.py --strategy donchian_breakout \
        --param entry_period=48 --param exit_period=24 --initial-cash 1000 \
        --vol-target 0.4 --stop-atr 3 --max-drawdown 0.25

Market data comes from Binance's public production endpoint (the testnet's own
candles are thin and erratic); orders go to the testnet only.  Restarting with
the same --run-id resumes the run and reconciles with testnet balances.
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
from src.core.engine import TradingEngine
from src.core.logging_setup import configure_logging
from src.data.interfaces import MarketDataProvider
from src.data.providers.binance_spot import BinanceSpotMarketDataProvider
from src.database.store import TradingStore
from src.execution.binance_testnet import (
    BinanceTestnetClient,
    BinanceTradingError,
    Credentials,
    TestnetBroker,
)
from src.paper.feed import ClosedBarFeed
from src.paper.reconciliation import reconcile_with_exchange
from src.paper.trader import LiveTrader, RunConflictError, TraderSettings
from src.strategies.registry import StrategyConfigurationError, strategy_factory

STATE_DIR = PROJECT_ROOT / "data" / "paper"
LOG_DIR = PROJECT_ROOT / "logs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_strategy_arguments(parser, "donchian_breakout")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--run-id", help="Name of the run to create or resume.")
    parser.add_argument("--db", type=Path, default=STATE_DIR / "trading.db")
    parser.add_argument("--kill-file", type=Path, default=STATE_DIR / "KILL")
    parser.add_argument("--warmup-bars", type=int, default=500)
    parser.add_argument("--once", action="store_true", help="Process available bars, then exit.")
    add_cost_arguments(parser)
    add_risk_arguments(parser)
    return parser


def main(
    arguments: Sequence[str] | None = None,
    *,
    client: BinanceTestnetClient | None = None,
    provider: MarketDataProvider | None = None,
) -> int:
    """Run the trader; ``client`` and ``provider`` can be injected for tests."""

    args = build_parser().parse_args(arguments)
    parameters = parse_key_values(args.param)
    run_id = args.run_id or f"testnet-{args.strategy}-{args.timeframe}"
    configure_logging(LOG_DIR / f"{run_id}.log")

    try:
        strategy_factory(args.strategy, parameters)
        if client is None:
            client = BinanceTestnetClient(Credentials.from_environment())
            client.sync_time()
        broker = TestnetBroker(client, args.symbol, run_tag=run_id)
        settings = TraderSettings(
            mode="testnet",
            symbol=args.symbol,
            timeframe=args.timeframe,
            strategy=args.strategy,
            parameters=parameters,
            initial_cash=args.initial_cash,
            cost_model=cost_model_from(args),
            risk_config=risk_config_from(args),
            engine_config=engine_config_from(args),
            warmup_bars=args.warmup_bars,
            backfill_missed_bars=False,
        )
        trader = LiveTrader(
            store=TradingStore(args.db),
            feed=ClosedBarFeed(
                provider or BinanceSpotMarketDataProvider(), args.symbol, args.timeframe
            ),
            broker=broker,
            settings=settings,
            run_id=run_id,
            heartbeat_path=LOG_DIR / f"{run_id}.heartbeat.json",
            kill_file=args.kill_file,
            on_start=lambda engine: _reconcile(trader, engine, broker),
        )
        trader.start()
    except (
        BinanceTradingError,
        RunConflictError,
        StrategyConfigurationError,
        ValueError,
        OSError,
    ) as error:
        print(f"Testnet trading failed to start: {error}", file=sys.stderr)
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


def _reconcile(trader: LiveTrader, engine: TradingEngine, broker: TestnetBroker) -> None:
    for warning in reconcile_with_exchange(engine, broker):
        trader.store.log_event(trader.run_id, "WARNING", warning)


if __name__ == "__main__":
    raise SystemExit(main())
