"""Trend-following strategies that read higher timeframes from the bars they are fed.

Both strategies run on any base timeframe that divides the higher ones (1h
bars by default).  A higher-timeframe bar counts as closed only when the base
bar ending exactly on its UTC boundary has closed, so no partial period is
ever used.
"""

from datetime import timedelta
from decimal import Decimal

from src.data.models import OHLCVCandle

from .base import FLAT, FULLY_LONG
from .indicators import ExponentialMovingAverage, SimpleMovingAverage

_ONE_MILLISECOND = timedelta(milliseconds=1)
_TREND_TIMEFRAME_MINUTES = (60, 240, 1_440)


class PeriodCloses:
    """Report the close of each completed ``minutes``-long UTC period.

    ``update`` returns the period's close when ``bar`` is its last bar, and
    ``None`` for every other bar.
    """

    def __init__(self, minutes: int) -> None:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        self._seconds = minutes * 60

    def update(self, bar: OHLCVCandle) -> float | None:
        period_end = bar.close_time + _ONE_MILLISECOND
        if int(period_end.timestamp()) % self._seconds:
            return None
        return float(bar.close)


class TrendFilter:
    """Long while the daily close is above its ``sma_days`` moving average, flat below.

    ``band`` adds hysteresis: entering needs a close above ``SMA * (1 + band)``
    and exiting a close below ``SMA * (1 - band)``, which avoids flipping on
    noise around the average.  Signals change only at UTC day closes.
    """

    def __init__(self, sma_days: int = 100, band: float = 0.0) -> None:
        if not 0 <= band < 1:
            raise ValueError("band must be between 0 and 1")
        self._daily = PeriodCloses(1_440)
        self._sma = SimpleMovingAverage(sma_days)
        self._band = band
        self._long = False

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        close = self._daily.update(bar)
        if close is None:
            return None
        sma = self._sma.update(close)
        if sma is None:
            return None

        if not self._long and close > sma * (1 + self._band):
            self._long = True
        elif self._long and close < sma * (1 - self._band):
            self._long = False
        return FULLY_LONG if self._long else FLAT


class MultiTimeframeTrend:
    """Hold one third of full exposure for each of 1h, 4h, and 1d that is in an uptrend.

    A timeframe is in an uptrend while its close is above its
    ``ema_period``-bar EMA.  With fewer than ``min_votes`` timeframes in an
    uptrend the strategy is flat; ``min_votes=3`` therefore trades only when
    all three agree.
    """

    def __init__(self, ema_period: int = 50, min_votes: int = 1) -> None:
        if not 1 <= min_votes <= len(_TREND_TIMEFRAME_MINUTES):
            raise ValueError(f"min_votes must be between 1 and {len(_TREND_TIMEFRAME_MINUTES)}")
        self._timeframes = [
            (PeriodCloses(minutes), ExponentialMovingAverage(ema_period))
            for minutes in _TREND_TIMEFRAME_MINUTES
        ]
        self._votes: list[bool | None] = [None] * len(self._timeframes)
        self._min_votes = min_votes

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        changed = False
        for index, (closes, average) in enumerate(self._timeframes):
            close = closes.update(bar)
            if close is None:
                continue
            ema = average.update(close)
            if ema is not None:
                self._votes[index] = close > ema
                changed = True
        if not changed or any(vote is None for vote in self._votes):
            return None

        votes = sum(1 for vote in self._votes if vote)
        if votes < self._min_votes:
            return FLAT
        return Decimal(votes) / Decimal(len(self._votes))
