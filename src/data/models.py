"""Normalized, provider-neutral market-data models."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class OHLCVCandle:
    """One time-bounded OHLCV observation for a market symbol."""

    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class CandleRequest:
    """A provider-neutral request for historical OHLCV candles."""

    symbol: str
    timeframe: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int | None = None
