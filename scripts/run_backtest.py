"""Backtest a strategy on a processed candle dataset.

Examples:
    python scripts/run_backtest.py --strategy buy_and_hold
    python scripts/run_backtest.py --strategy ma_crossover --param fast_period=24 \
        --param slow_period=96 --stop-atr 3 --max-drawdown 0.3

Orders are decided on a bar's close and filled at the next bar's open, with
fees and slippage charged on every fill.  Build datasets first with
scripts/build_dataset.py.
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
    add_data_arguments,
    add_risk_arguments,
    add_strategy_arguments,
    cost_model_from,
    engine_config_from,
    format_metrics,
    load_bars,
    parse_key_values,
    resolve_paths,
    risk_config_from,
    write_fills_csv,
)
from src.backtest.runner import run_backtest
from src.data.dataset import DatasetError
from src.strategies.registry import StrategyConfigurationError, strategy_factory

DEFAULT_DATA = "data/processed/btcusdt_1h.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_arguments(parser, DEFAULT_DATA)
    add_strategy_arguments(parser, "buy_and_hold")
    add_cost_arguments(parser)
    add_risk_arguments(parser)
    parser.add_argument("--fills-output", type=Path, help="Write every fill to this CSV.")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        factory = strategy_factory(args.strategy, parse_key_values(args.param))
        risk_config = risk_config_from(args)
        bars = load_bars(resolve_paths(args.data, PROJECT_ROOT), args.start, args.end)
        if len(bars) < 2:
            raise DatasetError("the selected period contains fewer than two bars")
        result = run_backtest(
            bars,
            factory,
            initial_cash=args.initial_cash,
            cost_model=cost_model_from(args),
            risk_config=risk_config,
            engine_config=engine_config_from(args),
        )
    except (
        argparse.ArgumentTypeError,
        DatasetError,
        FileNotFoundError,
        StrategyConfigurationError,
        ValueError,
    ) as error:
        print(f"Backtest failed: {error}", file=sys.stderr)
        return 1

    print(
        f"{args.strategy} on {len(bars):,} {bars[0].timeframe} bars, "
        f"{bars[0].open_time:%Y-%m-%d} to {bars[-1].open_time:%Y-%m-%d} UTC"
    )
    print(format_metrics(result.metrics))
    if result.halt_reason:
        print(f"  Trading halted: {result.halt_reason}")
    if args.fills_output:
        write_fills_csv(args.fills_output, result.fills)
        print(f"Fills written to {args.fills_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
