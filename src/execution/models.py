"""Orders and fills shared by simulated and real execution."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderReason(StrEnum):
    """Why an order was created; recorded on fills for later analysis."""

    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    RISK_LIMIT = "risk_limit"
    END_OF_PERIOD = "end_of_period"


@dataclass(frozen=True, slots=True)
class Order:
    """A market order for ``quantity`` base-asset units, decided at ``created_at``.

    ``quote_budget`` caps what a buy may spend including fees; if the price
    moved up before execution, the broker buys less rather than overspending.
    """

    side: Side
    quantity: Decimal
    created_at: datetime
    reason: OrderReason = OrderReason.SIGNAL
    client_order_id: str | None = None
    quote_budget: Decimal | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed trade.  ``fee`` is charged in the quote currency."""

    timestamp: datetime
    side: Side
    quantity: Decimal
    price: Decimal
    fee: Decimal
    reason: OrderReason = OrderReason.SIGNAL
    client_order_id: str | None = None

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price
