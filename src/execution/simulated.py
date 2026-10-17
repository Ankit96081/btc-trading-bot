"""A deterministic broker that fills orders against historical or live bars."""

from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from src.data.models import OHLCVCandle

from .costs import CostModel
from .models import Fill, Order, OrderReason, Side


class SimulatedBroker:
    """Fills market orders at the next bar's open, and stops inside the bar.

    An order decided at bar *t*'s close is queued and filled at bar *t+1*'s
    open, adjusted for slippage and charged fees, so a strategy can never
    trade at a price it used to make its decision.  A stop that is touched
    fills at the stop price, or at the bar's open if the market gapped below
    it, which is the realistic worse price.
    """

    def __init__(
        self, cost_model: CostModel | None = None, *, quantity_step: Decimal = Decimal("0.00001")
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.quantity_step = quantity_step
        self._pending: Order | None = None

    @property
    def pending_order(self) -> Order | None:
        return self._pending

    def on_bar(self, bar: OHLCVCandle) -> list[Fill]:
        order, self._pending = self._pending, None
        if order is None:
            return []
        fill = self._fill(order, bar.open, bar)
        return [] if fill is None else [fill]

    def submit(self, order: Order, bar: OHLCVCandle) -> list[Fill]:
        self._pending = order
        return []

    def stop_out(self, bar: OHLCVCandle, stop_price: Decimal, quantity: Decimal) -> list[Fill]:
        reference = min(bar.open, stop_price)
        order = Order(Side.SELL, quantity, bar.open_time, OrderReason.STOP_LOSS)
        fill = self._fill(order, reference, bar)
        return [] if fill is None else [fill]

    def close_at(self, bar: OHLCVCandle, quantity: Decimal, reason: OrderReason) -> list[Fill]:
        """Sell ``quantity`` at ``bar``'s close, e.g. to end a simulation flat."""

        order = Order(Side.SELL, quantity, bar.open_time, reason)
        fill = self._fill(order, bar.close, bar)
        return [] if fill is None else [fill]

    def cancel_pending(self) -> None:
        self._pending = None

    def snapshot(self) -> dict[str, str | None]:
        """Return the queued order, if any, so a restart does not lose it."""

        order = self._pending
        if order is None:
            return {}
        return {
            "side": order.side.value,
            "quantity": str(order.quantity),
            "created_at": order.created_at.isoformat(),
            "reason": order.reason.value,
            "client_order_id": order.client_order_id,
            "quote_budget": None if order.quote_budget is None else str(order.quote_budget),
        }

    def restore(self, snapshot: dict[str, str | None]) -> None:
        if not snapshot:
            self._pending = None
            return
        budget = snapshot.get("quote_budget")
        self._pending = Order(
            side=Side(snapshot["side"]),
            quantity=Decimal(str(snapshot["quantity"])),
            created_at=datetime.fromisoformat(str(snapshot["created_at"])),
            reason=OrderReason(snapshot["reason"]),
            client_order_id=snapshot.get("client_order_id"),
            quote_budget=None if budget is None else Decimal(budget),
        )

    def _fill(self, order: Order, reference_price: Decimal, bar: OHLCVCandle) -> Fill | None:
        price = self.cost_model.execution_price(order.side, reference_price)
        quantity = order.quantity
        if order.side is Side.BUY and order.quote_budget is not None:
            affordable = order.quote_budget / (price * (1 + self.cost_model.fee_rate))
            quantity = min(quantity, round_down(affordable, self.quantity_step))
        if quantity <= 0:
            return None
        return Fill(
            timestamp=bar.open_time,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=self.cost_model.fee(quantity * price),
            reason=order.reason,
            client_order_id=order.client_order_id,
        )


def round_down(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` down to a multiple of ``step`` (an exchange lot size)."""

    if value <= 0:
        return Decimal(0)
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step
