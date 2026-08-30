"""Interfaces used by the application to obtain normalized market data."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .models import CandleRequest, OHLCVCandle


@runtime_checkable
class MarketDataProvider(Protocol):
    """Fetch normalized market data without exposing provider-specific details.

    Future exchange adapters, including a Binance adapter, should implement this
    protocol and translate their provider payloads into ``OHLCVCandle`` objects.
    Application code should depend only on this protocol and the data models.
    """

    def get_candles(self, request: CandleRequest) -> Sequence[OHLCVCandle]:
        """Return candles matching ``request`` in ascending open-time order."""
