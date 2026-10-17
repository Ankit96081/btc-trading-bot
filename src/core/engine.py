"""The bar-by-bar trading loop shared by backtests, paper trading, and the testnet."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.data.models import OHLCVCandle
from src.execution.interfaces import Broker
from src.execution.models import Fill, Order, OrderReason, Side
from src.execution.simulated import round_down
from src.risk.manager import RiskManager
from src.strategies.base import FLAT, Strategy

from .portfolio import Portfolio


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Order-generation settings.

    ``min_rebalance_exposure`` suppresses small adjustments to an existing
    position (for example from volatility targeting) that would cost fees
    without meaningfully changing risk.  Entries from flat and full exits are
    never suppressed by it.
    """

    min_rebalance_exposure: Decimal = Decimal("0.05")
    min_order_notional: Decimal = Decimal(10)
    quantity_step: Decimal = Decimal("0.00001")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """Marked-to-market account state at a bar's close."""

    timestamp: datetime
    close: Decimal
    equity: Decimal
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One complete trade, from a flat position back to flat."""

    entry_time: datetime
    exit_time: datetime
    cost: Decimal
    proceeds: Decimal
    exit_reason: OrderReason

    @property
    def pnl(self) -> Decimal:
        return self.proceeds - self.cost

    @property
    def return_fraction(self) -> Decimal:
        return self.pnl / self.cost if self.cost else Decimal(0)


@dataclass(slots=True)
class _OpenTrip:
    entry_time: datetime
    cost: Decimal = Decimal(0)
    proceeds: Decimal = Decimal(0)


@dataclass(slots=True)
class EngineState:
    """Everything the engine has recorded so far."""

    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    round_trips: list[RoundTrip] = field(default_factory=list)


class TradingEngine:
    """Run a strategy through a broker under a risk manager, one closed bar at a time.

    For each closed bar the engine:

    1. applies fills the broker produced for earlier orders (at this bar's open),
    2. exits through the broker if the bar traded through the stop price,
    3. marks the portfolio to the bar's close and lets the risk manager react,
    4. asks the strategy for a target exposure, which risk limits then adjust,
    5. submits the order needed to reach that exposure.

    Bars passed with ``trading=False`` only warm up the strategy and risk
    indicators; nothing is traded or recorded.
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        portfolio: Portfolio,
        risk: RiskManager,
        config: EngineConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.config = config or EngineConfig()
        self.state = EngineState()
        self._strategy_target = FLAT
        self._open_trip: _OpenTrip | None = None

    def process_bar(self, bar: OHLCVCandle, *, trading: bool = True) -> None:
        if not trading:
            self._update_signals(bar)
            return

        self._apply_fills(self.broker.on_bar(bar))
        self._check_stop(bar)

        equity = self.portfolio.equity(bar.close)
        forced_exit = self.risk.update_equity(bar.open_time, equity)
        self.state.equity_curve.append(
            EquityPoint(bar.open_time, bar.close, equity, self.portfolio.quantity)
        )

        self._update_signals(bar)
        target = self.risk.target_exposure(self._strategy_target)
        reason = OrderReason.RISK_LIMIT if forced_exit or self.risk.halted else OrderReason.SIGNAL
        order = self._order_for_target(target, bar, equity, reason)
        if order is not None:
            self._apply_fills(self.broker.submit(order, bar))

    def snapshot(self) -> dict[str, object]:
        """Return the state needed to resume after a restart (see :meth:`restore`)."""

        trip = self._open_trip
        return {
            "cash": str(self.portfolio.cash),
            "quantity": str(self.portfolio.quantity),
            "strategy_target": str(self._strategy_target),
            "risk": self.risk.snapshot(),
            "open_trip": None
            if trip is None
            else {
                "entry_time": trip.entry_time.isoformat(),
                "cost": str(trip.cost),
                "proceeds": str(trip.proceeds),
            },
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore portfolio, risk-limit, and open-trade state saved by :meth:`snapshot`.

        Call this after replaying recent bars with ``trading=False``, which
        rebuilds strategy and indicator state that is not persisted.
        """

        self.portfolio.cash = Decimal(snapshot["cash"])
        self.portfolio.quantity = Decimal(snapshot["quantity"])
        self._strategy_target = Decimal(snapshot["strategy_target"])
        self.risk.restore(snapshot["risk"])
        trip = snapshot.get("open_trip")
        self._open_trip = (
            None
            if trip is None
            else _OpenTrip(
                entry_time=datetime.fromisoformat(trip["entry_time"]),
                cost=Decimal(trip["cost"]),
                proceeds=Decimal(trip["proceeds"]),
            )
        )

    def apply_fills(self, fills: list[Fill]) -> None:
        """Apply fills that happened outside the bar loop, e.g. recovered after a crash."""

        self._apply_fills(fills)

    def apply_closing_fills(self, fills: list[Fill], bar: OHLCVCandle) -> None:
        """Apply fills executed at ``bar``'s close and re-mark that bar's equity."""

        self._apply_fills(fills)
        if self.state.equity_curve and self.state.equity_curve[-1].timestamp == bar.open_time:
            self.state.equity_curve[-1] = EquityPoint(
                bar.open_time, bar.close, self.portfolio.equity(bar.close), self.portfolio.quantity
            )

    def _update_signals(self, bar: OHLCVCandle) -> None:
        self.risk.observe_bar(bar)
        signal = self.strategy.on_bar(bar)
        if signal is not None:
            self._strategy_target = signal

    def _check_stop(self, bar: OHLCVCandle) -> None:
        stop = self.risk.stop_price
        if stop is not None and self.portfolio.quantity > 0 and bar.low <= stop:
            self._apply_fills(self.broker.stop_out(bar, stop, self.portfolio.quantity))

    def _order_for_target(
        self, target: Decimal, bar: OHLCVCandle, equity: Decimal, reason: OrderReason
    ) -> Order | None:
        price = bar.close
        current = self.portfolio.quantity
        if equity <= 0 or price <= 0:
            return None

        desired = round_down(target * equity / price, self.config.quantity_step)
        delta = desired - current
        if delta == 0:
            return None
        notional_fraction = abs(delta) * price / equity

        if delta > 0:
            if abs(delta) * price < self.config.min_order_notional:
                return None
            if current > 0 and notional_fraction < self.config.min_rebalance_exposure:
                return None
            if not self.risk.can_add_risk(bar.open_time):
                return None
            return Order(
                Side.BUY, delta, bar.open_time, reason, quote_budget=self.portfolio.cash
            )

        if desired > 0 and (
            notional_fraction < self.config.min_rebalance_exposure
            or abs(delta) * price < self.config.min_order_notional
        ):
            return None
        return Order(Side.SELL, -delta, bar.open_time, reason)

    def _apply_fills(self, fills: list[Fill]) -> None:
        for fill in fills:
            before = self.portfolio.quantity
            self.portfolio.apply(fill)
            after = self.portfolio.quantity
            self.risk.record_fill(fill, before, after)
            self.state.fills.append(fill)
            self._track_round_trip(fill, before, after)

    def _track_round_trip(self, fill: Fill, before: Decimal, after: Decimal) -> None:
        if before == 0 and after > 0:
            self._open_trip = _OpenTrip(entry_time=fill.timestamp)
        trip = self._open_trip
        if trip is None:
            return
        if fill.side is Side.BUY:
            trip.cost += fill.notional + fill.fee
        else:
            trip.proceeds += fill.notional - fill.fee
        if after == 0:
            self.state.round_trips.append(
                RoundTrip(trip.entry_time, fill.timestamp, trip.cost, trip.proceeds, fill.reason)
            )
            self._open_trip = None
