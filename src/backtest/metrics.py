"""Performance statistics for an equity curve and its trades."""

import itertools
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

from src.core.engine import EquityPoint, RoundTrip
from src.data.timeframes import bars_per_year
from src.execution.models import Fill


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary statistics.  Returns and drawdowns are fractions (0.1 = 10%)."""

    bars: int
    days: float
    initial_equity: float
    final_equity: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    exposure_time: float
    fills: int
    fills_per_day: float
    round_trips: int
    win_rate: float
    profit_factor: float
    average_trade_return: float
    fees_paid: float
    turnover: float
    benchmark_return: float
    benchmark_sharpe_ratio: float
    benchmark_max_drawdown: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(
    equity_curve: Sequence[EquityPoint],
    fills: Sequence[Fill],
    round_trips: Sequence[RoundTrip],
    *,
    timeframe: str,
    initial_equity: Decimal,
    benchmark_start_price: Decimal,
) -> PerformanceMetrics:
    """Compute performance for a run whose equity was sampled at every bar close.

    The first period's return is measured from ``initial_equity`` (before the
    first bar), and the buy-and-hold benchmark from ``benchmark_start_price``
    (the first bar's open), so both cover exactly the same interval.  Sharpe
    and Sortino ratios assume a zero risk-free rate.
    """

    if not equity_curve:
        raise ValueError("equity_curve must not be empty")

    periods_per_year = bars_per_year(timeframe)
    days = len(equity_curve) * 365 / periods_per_year
    equities = [float(initial_equity)] + [float(point.equity) for point in equity_curve]
    prices = [float(benchmark_start_price)] + [float(point.close) for point in equity_curve]
    returns = _simple_returns(equities)
    benchmark_returns = _simple_returns(prices)

    total_return = equities[-1] / equities[0] - 1
    annualized_return = _annualize(total_return, len(returns), periods_per_year)
    max_drawdown = _max_drawdown(equities)

    winners = [trip.pnl for trip in round_trips if trip.pnl > 0]
    losers = [trip.pnl for trip in round_trips if trip.pnl <= 0]
    gross_loss = -sum(losers, Decimal(0))
    traded_notional = sum((fill.notional for fill in fills), Decimal(0))
    average_equity = sum(equities) / len(equities)

    return PerformanceMetrics(
        bars=len(equity_curve),
        days=days,
        initial_equity=equities[0],
        final_equity=equities[-1],
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=_stdev(returns) * math.sqrt(periods_per_year),
        sharpe_ratio=_sharpe(returns, periods_per_year),
        sortino_ratio=_sortino(returns, periods_per_year),
        max_drawdown=max_drawdown,
        calmar_ratio=annualized_return / max_drawdown if max_drawdown > 0 else 0.0,
        exposure_time=sum(1 for point in equity_curve if point.quantity > 0) / len(equity_curve),
        fills=len(fills),
        fills_per_day=len(fills) / days if days else 0.0,
        round_trips=len(round_trips),
        win_rate=len(winners) / len(round_trips) if round_trips else 0.0,
        profit_factor=(
            float(sum(winners, Decimal(0)) / gross_loss)
            if gross_loss > 0
            else (math.inf if winners else 0.0)
        ),
        average_trade_return=(
            float(sum((trip.return_fraction for trip in round_trips), Decimal(0)))
            / len(round_trips)
            if round_trips
            else 0.0
        ),
        fees_paid=float(sum((fill.fee for fill in fills), Decimal(0))),
        turnover=float(traded_notional) / average_equity if average_equity else 0.0,
        benchmark_return=prices[-1] / prices[0] - 1,
        benchmark_sharpe_ratio=_sharpe(benchmark_returns, periods_per_year),
        benchmark_max_drawdown=_max_drawdown(prices),
    )


def _simple_returns(values: Sequence[float]) -> list[float]:
    return [
        current / previous - 1 if previous else 0.0
        for previous, current in itertools.pairwise(values)
    ]


def _annualize(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods == 0 or total_return <= -1:
        return -1.0 if total_return <= -1 else 0.0
    return (1 + total_return) ** (periods_per_year / periods) - 1


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _sharpe(returns: Sequence[float], periods_per_year: float) -> float:
    deviation = _stdev(returns)
    return _mean(returns) / deviation * math.sqrt(periods_per_year) if deviation else 0.0


def _sortino(returns: Sequence[float], periods_per_year: float) -> float:
    if not returns:
        return 0.0
    downside = math.sqrt(sum(min(value, 0.0) ** 2 for value in returns) / len(returns))
    return _mean(returns) / downside * math.sqrt(periods_per_year) if downside else 0.0


def _max_drawdown(values: Sequence[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, 1 - value / peak)
    return worst
