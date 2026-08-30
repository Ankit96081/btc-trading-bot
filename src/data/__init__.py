"""Provider-neutral market-data contracts and validation utilities."""

from .historical_downloader import (
    ExistingOutputError,
    HistoricalDataDownloader,
    HistoricalDataDownloadError,
    HistoricalDataIntegrityError,
    HistoricalDownloadResult,
)
from .interfaces import MarketDataProvider
from .models import CandleRequest, OHLCVCandle
from .validation import CandleValidationError, validate_candle, validate_candles

__all__ = [
    "CandleRequest",
    "CandleValidationError",
    "ExistingOutputError",
    "HistoricalDataDownloadError",
    "HistoricalDataDownloader",
    "HistoricalDataIntegrityError",
    "HistoricalDownloadResult",
    "MarketDataProvider",
    "OHLCVCandle",
    "validate_candle",
    "validate_candles",
]
