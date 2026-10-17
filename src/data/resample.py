"""Aggregate candles into a larger timeframe, emitting only complete bars."""

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from .models import OHLCVCandle
from .timeframes import timeframe_to_minutes, timeframe_to_timedelta

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ONE_MILLISECOND = timedelta(milliseconds=1)


class ResampleError(ValueError):
    """Raised when candles cannot be aggregated into the requested timeframe."""


def resample_candles(candles: Iterable[OHLCVCandle], timeframe: str) -> tuple[OHLCVCandle, ...]:
    """Aggregate ordered candles into ``timeframe`` bars aligned to the UTC epoch.

    A target bar is emitted only when every source bar inside it is present.
    Buckets touched by a data gap, and a trailing bucket that has not finished
    yet, are dropped rather than emitted with partial data.
    """

    aggregator = BarAggregator(timeframe)
    bars: list[OHLCVCandle] = []
    for candle in candles:
        completed = aggregator.add(candle)
        if completed is not None:
            bars.append(completed)
    return tuple(bars)


class BarAggregator:
    """Incrementally aggregate source candles into complete target-timeframe bars.

    The same aggregator serves historical resampling and live feeds: feed it
    source candles in order and it returns each target bar as soon as the last
    source candle of that bar arrives.
    """

    def __init__(self, timeframe: str) -> None:
        self._timeframe = timeframe
        self._target_minutes = timeframe_to_minutes(timeframe)
        self._target_duration = timeframe_to_timedelta(timeframe)
        self._bucket: list[OHLCVCandle] = []
        self._bucket_start: datetime | None = None
        self._source_timeframe: str | None = None
        self._expected_count = 0
        self._last_open_time: datetime | None = None

    def add(self, candle: OHLCVCandle) -> OHLCVCandle | None:
        """Add one source candle; return a completed target bar, if any."""

        self._check_source(candle)
        bucket_start = _bucket_start(candle.open_time, self._target_duration)
        if bucket_start != self._bucket_start:
            self._bucket = []
            self._bucket_start = bucket_start
        self._bucket.append(candle)

        if len(self._bucket) < self._expected_count:
            return None

        bar = _aggregate(self._bucket, self._timeframe, bucket_start, self._target_duration)
        self._bucket = []
        self._bucket_start = None
        return bar

    def _check_source(self, candle: OHLCVCandle) -> None:
        if self._source_timeframe is None:
            source_minutes = timeframe_to_minutes(candle.timeframe)
            if self._target_minutes < source_minutes or self._target_minutes % source_minutes:
                raise ResampleError(
                    f"cannot resample {candle.timeframe} candles into {self._timeframe}"
                )
            self._source_timeframe = candle.timeframe
            self._expected_count = self._target_minutes // source_minutes
        elif candle.timeframe != self._source_timeframe:
            raise ResampleError("all source candles must share one timeframe")

        open_time = candle.open_time.astimezone(UTC)
        if self._last_open_time is not None and open_time <= self._last_open_time:
            raise ResampleError("source candles must be in strictly increasing time order")
        self._last_open_time = open_time


def _bucket_start(open_time: datetime, duration: timedelta) -> datetime:
    elapsed = open_time.astimezone(UTC) - _EPOCH
    return _EPOCH + (elapsed // duration) * duration


def _aggregate(
    bucket: list[OHLCVCandle], timeframe: str, bucket_start: datetime, duration: timedelta
) -> OHLCVCandle:
    first = bucket[0]
    return OHLCVCandle(
        symbol=first.symbol,
        timeframe=timeframe,
        open_time=bucket_start,
        close_time=bucket_start + duration - _ONE_MILLISECOND,
        open=first.open,
        high=max(candle.high for candle in bucket),
        low=min(candle.low for candle in bucket),
        close=bucket[-1].close,
        volume=sum((candle.volume for candle in bucket), start=Decimal(0)),
    )
