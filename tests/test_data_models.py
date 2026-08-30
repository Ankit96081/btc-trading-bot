"""Tests for provider-neutral market-data models."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.data.models import CandleRequest, OHLCVCandle


def test_ohlcv_candle_preserves_normalized_values() -> None:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    candle = OHLCVCandle(
        symbol="BTC/USDT",
        timeframe="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1),
        open=Decimal("65000.10"),
        high=Decimal("65025.00"),
        low=Decimal("64990.50"),
        close=Decimal("65020.75"),
        volume=Decimal("1.250"),
    )

    assert candle.symbol == "BTC/USDT"
    assert candle.close == Decimal("65020.75")
    assert candle.volume == Decimal("1.250")


def test_candle_request_can_represent_a_bounded_history_query() -> None:
    start_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    request = CandleRequest(
        symbol="BTC/USDT",
        timeframe="1h",
        start_time=start_time,
        end_time=start_time + timedelta(hours=24),
        limit=24,
    )

    assert request.symbol == "BTC/USDT"
    assert request.limit == 24


def test_candles_are_immutable() -> None:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    candle = OHLCVCandle(
        symbol="BTC/USDT",
        timeframe="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1),
        open=Decimal(1),
        high=Decimal(2),
        low=Decimal(1),
        close=Decimal(2),
        volume=Decimal(3),
    )

    with pytest.raises(AttributeError):
        candle.close = Decimal(4)
