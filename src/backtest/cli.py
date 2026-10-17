"""Argument parsing and reporting helpers shared by the research scripts."""

import argparse
import csv
import glob
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.core.engine import EngineConfig
from src.data.csv_format import (
    CANDLE_CSV_COLUMNS,
    candle_to_csv_row,
    format_utc_timestamp,
)
from src.data.dataset import load_candles
from src.data.models import OHLCVCandle
from src.execution.costs import CostModel
from src.execution.models import Fill
from src.risk.manager import RiskConfig

from .metrics import PerformanceMetrics


def add_data_arguments(parser: argparse.ArgumentParser, default_data: str) -> None:
    parser.add_argument(
        "--data",
        nargs="+",
        default=[default_data],
        metavar="PATH_OR_GLOB",
        help=f"Candle CSV files or glob patterns (default: {default_data}).",
    )
    parser.add_argument("--start", type=parse_utc_timestamp, help="Inclusive UTC start time.")
    parser.add_argument("--end", type=parse_utc_timestamp, help="Exclusive UTC end time.")


def add_cost_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = CostModel()
    parser.add_argument("--initial-cash", type=positive_decimal, default=Decimal(10000))
    parser.add_argument(
        "--fee-rate",
        type=non_negative_decimal,
        default=defaults.fee_rate,
        help=f"Fee per fill as a fraction of notional (default: {defaults.fee_rate}).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=non_negative_decimal,
        default=defaults.slippage_bps,
        help=f"Adverse slippage per fill in basis points (default: {defaults.slippage_bps}).",
    )
    parser.add_argument(
        "--min-rebalance",
        type=non_negative_decimal,
        default=EngineConfig().min_rebalance_exposure,
        help="Ignore position adjustments smaller than this fraction of equity.",
    )


def add_risk_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("risk limits (all disabled unless given)")
    group.add_argument("--max-exposure", type=positive_decimal, default=Decimal(1))
    group.add_argument("--vol-target", type=float, help="Annualized volatility target, e.g. 0.4.")
    group.add_argument("--vol-lookback", type=int, default=168, help="Bars for volatility.")
    group.add_argument("--stop-atr", type=float, help="Stop-loss distance in ATR multiples.")
    group.add_argument("--atr-period", type=int, default=14)
    group.add_argument("--stop-pct", type=positive_decimal, help="Stop-loss distance as a fraction.")
    group.add_argument("--max-daily-loss", type=positive_decimal, help="Fraction, e.g. 0.03.")
    group.add_argument("--max-drawdown", type=positive_decimal, help="Kill switch, e.g. 0.25.")
    group.add_argument("--max-trades-per-day", type=int)


def add_strategy_arguments(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--strategy", default=default, help=f"Strategy name (default: {default}).")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Strategy parameter; repeat for several.",
    )


def cost_model_from(args: argparse.Namespace) -> CostModel:
    return CostModel(fee_rate=args.fee_rate, slippage_bps=args.slippage_bps)


def engine_config_from(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(min_rebalance_exposure=args.min_rebalance)


def risk_config_from(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(
        max_exposure=args.max_exposure,
        volatility_target=args.vol_target,
        volatility_lookback=args.vol_lookback,
        stop_loss_atr_multiple=args.stop_atr,
        stop_loss_fraction=args.stop_pct,
        atr_period=args.atr_period,
        max_daily_loss=args.max_daily_loss,
        max_drawdown=args.max_drawdown,
        max_trades_per_day=args.max_trades_per_day,
    )


def parse_key_values(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {item!r}")
        parsed[key.strip()] = value.strip()
    return parsed


def resolve_paths(patterns: Sequence[str], project_root: Path) -> list[Path]:
    """Expand glob patterns relative to the project root; plain paths must exist."""

    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        absolute = candidate if candidate.is_absolute() else project_root / candidate
        matches = sorted(Path(match) for match in glob.glob(str(absolute)))
        if not matches:
            raise FileNotFoundError(f"no files match {pattern}")
        paths.extend(matches)
    return paths


def load_bars(
    paths: Sequence[Path], start: datetime | None, end: datetime | None
) -> tuple[OHLCVCandle, ...]:
    """Load a dataset (gaps allowed) and trim it to ``[start, end)``."""

    bars = load_candles(paths, allow_gaps=True)
    return tuple(
        bar
        for bar in bars
        if (start is None or bar.open_time >= start) and (end is None or bar.open_time < end)
    )


def write_candles_csv(path: Path, candles: Sequence[OHLCVCandle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CANDLE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(candle_to_csv_row(candle) for candle in candles)
    temporary.replace(path)


def write_fills_csv(path: Path, fills: Sequence[Fill]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["timestamp", "side", "quantity", "price", "fee", "reason"])
        for fill in fills:
            writer.writerow(
                [
                    format_utc_timestamp(fill.timestamp),
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    fill.fee,
                    fill.reason.value,
                ]
            )


def format_metrics(metrics: PerformanceMetrics) -> str:
    rows = [
        ("Period (days)", f"{metrics.days:,.1f}"),
        ("Final equity", f"{metrics.final_equity:,.2f}"),
        ("Total return", _percent(metrics.total_return)),
        ("Annualized return", _percent(metrics.annualized_return)),
        ("Annualized volatility", _percent(metrics.annualized_volatility)),
        ("Sharpe ratio", f"{metrics.sharpe_ratio:.2f}"),
        ("Sortino ratio", f"{metrics.sortino_ratio:.2f}"),
        ("Max drawdown", _percent(metrics.max_drawdown)),
        ("Calmar ratio", f"{metrics.calmar_ratio:.2f}"),
        ("Time in market", _percent(metrics.exposure_time)),
        ("Fills (per day)", f"{metrics.fills} ({metrics.fills_per_day:.2f})"),
        ("Round trips", str(metrics.round_trips)),
        ("Win rate", _percent(metrics.win_rate)),
        ("Profit factor", f"{metrics.profit_factor:.2f}"),
        ("Avg trade return", _percent(metrics.average_trade_return)),
        ("Fees paid", f"{metrics.fees_paid:,.2f}"),
        ("Turnover (x equity)", f"{metrics.turnover:.1f}"),
        ("Buy & hold return", _percent(metrics.benchmark_return)),
        ("Buy & hold Sharpe", f"{metrics.benchmark_sharpe_ratio:.2f}"),
        ("Buy & hold max DD", _percent(metrics.benchmark_max_drawdown)),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label:<{width}}  {value}" for label, value in rows)


def parse_utc_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("must be UTC, e.g. 2024-01-01T00:00:00Z")
    return timestamp.astimezone(UTC)


def positive_decimal(value: str) -> Decimal:
    number = _decimal(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def non_negative_decimal(value: str) -> Decimal:
    number = _decimal(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def _decimal(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not number.is_finite():
        raise argparse.ArgumentTypeError("must be finite")
    return number


def _percent(value: float) -> str:
    return f"{value * 100:,.2f}%"
