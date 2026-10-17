"""The execution contract shared by the simulated broker and exchange clients."""

from decimal import Decimal
from typing import Protocol, runtime_checkable

from src.data.models import OHLCVCandle

from .models import Fill, Order, OrderReason


@runtime_checkable
class Broker(Protocol):
    """Executes the engine's orders.

    The engine calls these in a fixed order for every closed bar:
    ``on_bar`` (fills for work queued earlier), ``stop_out`` if a stop was
    touched, and finally ``submit`` for the order decided at the bar's close.
    A simulated broker fills submitted orders at the *next* bar's open; an
    exchange client fills them immediately.  Either way the engine applies
    whatever fills are returned.
    """

    def on_bar(self, bar: OHLCVCandle) -> list[Fill]:
        """Process a newly closed bar and return fills for previously queued orders."""

    def submit(self, order: Order, bar: OHLCVCandle) -> list[Fill]:
        """Submit an order decided at ``bar``'s close; return any immediate fills."""

    def stop_out(self, bar: OHLCVCandle, stop_price: Decimal, quantity: Decimal) -> list[Fill]:
        """Sell ``quantity`` because ``bar`` traded at or below ``stop_price``."""

    def close_at(self, bar: OHLCVCandle, quantity: Decimal, reason: OrderReason) -> list[Fill]:
        """Sell ``quantity`` now; ``bar`` is the latest closed bar (the simulated price)."""
