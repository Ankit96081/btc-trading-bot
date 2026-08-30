"""Tests for normalized OHLCV validation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.data.models import OHLCVCandle
from src.data.validation import CandleValidationError, validate_candle, validate_candles


def make_candle(**changes: object) -> OHLCVCandle:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "open_time": open_time,
        "close_time": open_time + timedelta(minutes=1),
        "open": Decimal(65000),
        "high": Decimal(65020),
        "low": Decimal(64980),
        "close": Decimal(65010),
        "volume": Decimal("1.5"),
    }
    values.update(changes)
    return OHLCVCandle(**values)  # type: ignore[arg-type]


def test_validate_candle_returns_a_valid_candle() -> None:
    candle = make_candle()

    assert validate_candle(candle) is candle


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"symbol": " "}, "symbol must be a non-empty string"),
        ({"volume": Decimal("-0.01")}, "volume must be greater than or equal to zero"),
        ({"high": Decimal(64999)}, "high cannot be lower than the open or close price"),
        ({"low": Decimal(65001)}, "low cannot exceed the open or close price"),
        ({"open": 65000}, "open must be a Decimal"),
    ],
)
def test_validate_candle_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(CandleValidationError, match=message):
        validate_candle(make_candle(**changes))


def test_validate_candle_rejects_naive_timestamps() -> None:
    naive_open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(CandleValidationError, match="open_time must be timezone-aware"):
        validate_candle(make_candle(open_time=naive_open_time))


def test_validate_candles_rejects_overlapping_time_ranges() -> None:
    first = make_candle()
    second = make_candle(
        open_time=first.open_time + timedelta(seconds=30),
        close_time=first.close_time + timedelta(seconds=30),
    )

    with pytest.raises(CandleValidationError, match="candles must not overlap"):
        validate_candles((first, second))


def test_validate_candles_rejects_mixed_symbols() -> None:
    first = make_candle()
    second = make_candle(
        symbol="ETH/USDT",
        open_time=first.close_time,
        close_time=first.close_time + timedelta(minutes=1),
    )

    with pytest.raises(CandleValidationError, match="all candles must have the same symbol"):
        validate_candles((first, second))


def test_validate_candles_returns_an_immutable_sequence() -> None:
    first = make_candle()
    second = make_candle(
        open_time=first.close_time,
        close_time=first.close_time + timedelta(minutes=1),
    )

    assert validate_candles((first, second)) == (first, second)
