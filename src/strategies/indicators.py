"""Incremental technical indicators.

Each indicator is fed one value (or bar) at a time with ``update`` and returns
``None`` until it has seen enough data.  Indicators work in ``float``: they
drive signals, not accounting, so exact decimal arithmetic is not needed and
floats keep parameter sweeps fast.
"""

import math
from collections import deque


class SimpleMovingAverage:
    """Arithmetic mean of the last ``period`` values."""

    def __init__(self, period: int) -> None:
        self.period = _require_period(period)
        self._values: deque[float] = deque()
        self._total = 0.0

    def update(self, value: float) -> float | None:
        self._values.append(value)
        self._total += value
        if len(self._values) > self.period:
            self._total -= self._values.popleft()
        if len(self._values) < self.period:
            return None
        return self._total / self.period


class ExponentialMovingAverage:
    """EMA with ``alpha = 2 / (period + 1)``, seeded with the first ``period`` values' SMA."""

    def __init__(self, period: int) -> None:
        self.period = _require_period(period)
        self._alpha = 2.0 / (period + 1)
        self._seed = SimpleMovingAverage(period)
        self._value: float | None = None

    def update(self, value: float) -> float | None:
        if self._value is None:
            self._value = self._seed.update(value)
        else:
            self._value += self._alpha * (value - self._value)
        return self._value


class RollingStandardDeviation:
    """Sample standard deviation of the last ``period`` values."""

    def __init__(self, period: int) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        self.period = period
        self._values: deque[float] = deque(maxlen=period)

    def update(self, value: float) -> float | None:
        self._values.append(value)
        if len(self._values) < self.period:
            return None
        mean = sum(self._values) / self.period
        variance = sum((item - mean) ** 2 for item in self._values) / (self.period - 1)
        return math.sqrt(variance)


class RollingMaximum:
    """Maximum of the last ``period`` values, using a monotonic queue."""

    def __init__(self, period: int) -> None:
        self.period = _require_period(period)
        self._window: deque[tuple[int, float]] = deque()
        self._count = 0

    def update(self, value: float) -> float | None:
        while self._window and self._window[-1][1] <= value:
            self._window.pop()
        self._window.append((self._count, value))
        if self._window[0][0] <= self._count - self.period:
            self._window.popleft()
        self._count += 1
        if self._count < self.period:
            return None
        return self._window[0][1]


class RollingMinimum:
    """Minimum of the last ``period`` values."""

    def __init__(self, period: int) -> None:
        self._maximum = RollingMaximum(period)
        self.period = period

    def update(self, value: float) -> float | None:
        negated = self._maximum.update(-value)
        return None if negated is None else -negated


class AverageTrueRange:
    """Wilder's ATR over ``period`` bars."""

    def __init__(self, period: int) -> None:
        self.period = _require_period(period)
        self._previous_close: float | None = None
        self._seed: list[float] = []
        self._value: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self._previous_close is None:
            true_range = high - low
        else:
            true_range = max(
                high - low, abs(high - self._previous_close), abs(low - self._previous_close)
            )
        self._previous_close = close

        if self._value is None:
            self._seed.append(true_range)
            if len(self._seed) == self.period:
                self._value = sum(self._seed) / self.period
                self._seed = []
        else:
            self._value = (self._value * (self.period - 1) + true_range) / self.period
        return self._value


class RelativeStrengthIndex:
    """Wilder's RSI over ``period`` price changes, in the range 0-100."""

    def __init__(self, period: int) -> None:
        self.period = _require_period(period)
        self._previous: float | None = None
        self._gains: list[float] = []
        self._losses: list[float] = []
        self._average_gain: float | None = None
        self._average_loss = 0.0

    def update(self, value: float) -> float | None:
        if self._previous is None:
            self._previous = value
            return None
        change = value - self._previous
        self._previous = value
        gain, loss = max(change, 0.0), max(-change, 0.0)

        if self._average_gain is None:
            self._gains.append(gain)
            self._losses.append(loss)
            if len(self._gains) < self.period:
                return None
            self._average_gain = sum(self._gains) / self.period
            self._average_loss = sum(self._losses) / self.period
        else:
            self._average_gain = (self._average_gain * (self.period - 1) + gain) / self.period
            self._average_loss = (self._average_loss * (self.period - 1) + loss) / self.period

        if self._average_loss == 0:
            return 100.0 if self._average_gain > 0 else 50.0
        relative_strength = self._average_gain / self._average_loss
        return 100.0 - 100.0 / (1.0 + relative_strength)


def _require_period(period: int) -> int:
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")
    return period
