"""Run a strategy over historical bars."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.core.engine import EngineConfig, EquityPoint, RoundTrip, TradingEngine
from src.core.portfolio import Portfolio
from src.data.models import OHLCVCandle
from src.execution.costs import CostModel
from src.execution.models import Fill, OrderReason
from src.execution.simulated import SimulatedBroker
from src.risk.manager import RiskConfig, RiskManager
from src.strategies.base import StrategyFactory

from .metrics import PerformanceMetrics, compute_metrics


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything a backtest produced.  Only bars after the warm-up are traded."""

    timeframe: str
    initial_cash: Decimal
    first_open: Decimal
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[Fill, ...]
    round_trips: tuple[RoundTrip, ...]
    final_quantity: Decimal
    halt_reason: str | None

    @property
    def metrics(self) -> PerformanceMetrics:
        return compute_metrics(
            self.equity_curve,
            self.fills,
            self.round_trips,
            timeframe=self.timeframe,
            initial_equity=self.initial_cash,
            benchmark_start_price=self.first_open,
        )


def run_backtest(
    bars: Sequence[OHLCVCandle],
    strategy_factory: StrategyFactory,
    *,
    initial_cash: Decimal = Decimal(10000),
    cost_model: CostModel | None = None,
    risk_config: RiskConfig | None = None,
    engine_config: EngineConfig | None = None,
    warmup_bars: int = 0,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """Backtest a fresh strategy over ``bars``.

    The first ``warmup_bars`` bars only prime indicators, which lets a test
    window start with fully formed signals without trading on history that
    belongs to another window.  Any order still pending after the last bar is
    discarded.  An open position is reported as-is, or, with
    ``liquidate_at_end``, sold at the last close with fees and slippage.
    """

    if not 0 <= warmup_bars < len(bars):
        raise ValueError("warmup_bars must leave at least one bar to trade")

    config = engine_config or EngineConfig()
    timeframe = bars[0].timeframe
    broker = SimulatedBroker(cost_model, quantity_step=config.quantity_step)
    engine = TradingEngine(
        strategy=strategy_factory(),
        broker=broker,
        portfolio=Portfolio(cash=initial_cash),
        risk=RiskManager(risk_config, timeframe),
        config=config,
    )
    for index, bar in enumerate(bars):
        engine.process_bar(bar, trading=index >= warmup_bars)
    if liquidate_at_end and engine.portfolio.quantity > 0:
        engine.apply_closing_fills(
            broker.close_at(bars[-1], engine.portfolio.quantity, OrderReason.END_OF_PERIOD),
            bars[-1],
        )

    return BacktestResult(
        timeframe=timeframe,
        initial_cash=initial_cash,
        first_open=bars[warmup_bars].open,
        equity_curve=tuple(engine.state.equity_curve),
        fills=tuple(engine.state.fills),
        round_trips=tuple(engine.state.round_trips),
        final_quantity=engine.portfolio.quantity,
        halt_reason=engine.risk.halt_reason,
    )
