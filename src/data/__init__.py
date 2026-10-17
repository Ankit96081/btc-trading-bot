"""Provider-neutral market-data contracts and validation utilities."""

from .dataset import DatasetError, find_gaps, load_candles
from .historical_downloader import (
    ExistingOutputError,
    HistoricalDataDownloader,
    HistoricalDataDownloadError,
    HistoricalDataIntegrityError,
    HistoricalDownloadResult,
)
from .interfaces import MarketDataProvider
from .models import CandleRequest, OHLCVCandle
from .resample import BarAggregator, ResampleError, resample_candles
from .timeframes import bars_per_year, timeframe_to_minutes, timeframe_to_timedelta
from .validation import CandleValidationError, validate_candle, validate_candles

__all__ = [
    "BarAggregator",
    "CandleRequest",
    "CandleValidationError",
    "DatasetError",
    "ExistingOutputError",
    "HistoricalDataDownloadError",
    "HistoricalDataDownloader",
    "HistoricalDataIntegrityError",
    "HistoricalDownloadResult",
    "MarketDataProvider",
    "OHLCVCandle",
    "ResampleError",
    "bars_per_year",
    "find_gaps",
    "load_candles",
    "resample_candles",
    "timeframe_to_minutes",
    "timeframe_to_timedelta",
    "validate_candle",
    "validate_candles",
]
