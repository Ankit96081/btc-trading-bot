"""Provider-neutral market-data contracts and validation utilities."""

from .interfaces import MarketDataProvider
from .models import CandleRequest, OHLCVCandle
from .validation import CandleValidationError, validate_candle, validate_candles

__all__ = [
    "CandleRequest",
    "CandleValidationError",
    "MarketDataProvider",
    "OHLCVCandle",
    "validate_candle",
    "validate_candles",
]
