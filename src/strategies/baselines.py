"""Deliberately simple reference strategies.

These are yardsticks, not candidates for live trading: any research strategy
must beat them out-of-sample after costs to be worth further work.
"""

from decimal import Decimal

from src.data.models import OHLCVCandle

from .base import FLAT, FULLY_LONG
from .indicators import (
    ExponentialMovingAverage,
    RollingMaximum,
    RollingMinimum,
    SimpleMovingAverage,
)


class BuyAndHold:
    """Fully long from the first bar onwards; the passive benchmark."""

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        return FULLY_LONG


class MovingAverageCrossover:
    """Long while the fast moving average is above the slow one, flat otherwise."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50, *, exponential: bool = False) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be shorter than slow_period")
        average = ExponentialMovingAverage if exponential else SimpleMovingAverage
        self._fast = average(fast_period)
        self._slow = average(slow_period)

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        close = float(bar.close)
        fast = self._fast.update(close)
        slow = self._slow.update(close)
        if fast is None or slow is None:
            return None
        return FULLY_LONG if fast > slow else FLAT


class DonchianBreakout:
    """Enter when the close breaks the prior ``entry_period`` high; exit below the prior ``exit_period`` low.

    Channels are computed from bars *before* the current one, so a bar is
    compared against history that excludes itself.
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10) -> None:
        self._highs = RollingMaximum(entry_period)
        self._lows = RollingMinimum(exit_period)
        self._upper: float | None = None
        self._lower: float | None = None
        self._long = False

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        close = float(bar.close)
        upper, lower = self._upper, self._lower
        self._upper = self._highs.update(float(bar.high))
        self._lower = self._lows.update(float(bar.low))
        if upper is None or lower is None:
            return None

        if not self._long and close > upper:
            self._long = True
        elif self._long and close < lower:
            self._long = False
        return FULLY_LONG if self._long else FLAT
