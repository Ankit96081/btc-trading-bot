"""The contract between trading strategies and the engines that run them."""

from collections.abc import Callable
from decimal import Decimal
from typing import Protocol, runtime_checkable

from src.data.models import OHLCVCandle

FLAT = Decimal(0)
FULLY_LONG = Decimal(1)


@runtime_checkable
class Strategy(Protocol):
    """A stateful signal generator fed one *closed* bar at a time.

    ``on_bar`` returns the desired exposure as a fraction of equity, from 0
    (flat) to 1 (fully long; spot trading is long-only), or ``None`` to keep
    the current target.  A strategy only ever sees bars that have closed, so
    it cannot look ahead; the engine executes its decision no earlier than the
    next bar's open.  Position sizing and risk limits are applied afterwards
    by the risk layer, so strategies express intent, not order sizes.
    """

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        """Consume one closed bar and return a target exposure or ``None``."""


StrategyFactory = Callable[[], Strategy]
"""Builds a fresh strategy instance; engines need new state for every run."""
