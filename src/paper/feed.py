"""Closed-bar polling on top of any market-data provider."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.data.interfaces import MarketDataProvider
from src.data.models import CandleRequest, OHLCVCandle
from src.data.timeframes import timeframe_to_timedelta

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ClosedBarFeed:
    """Return only bars that have fully closed, in order, without repeats.

    Exchanges also return the bar that is still forming; trading on it would
    be look-ahead, so any bar whose close time is not yet in the past is
    ignored until a later poll.  After downtime, one poll returns every bar
    that closed in the meantime.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        symbol: str,
        timeframe: str,
        *,
        clock: Clock = utc_now,
        settle_delay: timedelta = timedelta(seconds=5),
    ) -> None:
        self._provider = provider
        self.symbol = symbol
        self.timeframe = timeframe
        self._duration = timeframe_to_timedelta(timeframe)
        self._clock = clock
        self._settle_delay = settle_delay

    def now(self) -> datetime:
        return self._clock()

    def history(self, count: int) -> list[OHLCVCandle]:
        """Return up to ``count`` of the most recent closed bars."""

        if count <= 0:
            return []
        now = self._clock()
        bars = self._fetch(now - (count + 2) * self._duration, now, count + 2)
        return bars[-count:]

    def poll(self, after: datetime) -> list[OHLCVCandle]:
        """Return closed bars that opened after ``after``."""

        now = self._clock()
        start = after + self._duration
        if start >= now:
            return []
        expected = int((now - start) / self._duration) + 1
        return [bar for bar in self._fetch(start, now, expected) if bar.open_time > after]

    def bars_closed_since(self, open_time: datetime) -> int:
        """Number of bars that have closed after the bar that opened at ``open_time``."""

        elapsed = self._clock() - self._settle_delay - (open_time + self._duration)
        return max(0, int(elapsed / self._duration))

    def seconds_until_next_close(self, last_open_time: datetime) -> float:
        """Seconds until the bar after ``last_open_time`` has closed and settled."""

        next_close = last_open_time + 2 * self._duration + self._settle_delay
        return max(0.0, (next_close - self._clock()).total_seconds())

    def _fetch(self, start: datetime, now: datetime, limit: int) -> list[OHLCVCandle]:
        request = CandleRequest(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=start,
            end_time=now,
            limit=max(1, limit),
        )
        closed_before = now - self._settle_delay
        return [
            bar
            for bar in self._provider.get_candles(request)
            if bar.close_time < closed_before
        ]
