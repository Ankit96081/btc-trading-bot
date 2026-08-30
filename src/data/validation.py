"""Validation for normalized OHLCV market data."""

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from itertools import pairwise

from .models import OHLCVCandle


class CandleValidationError(ValueError):
    """Raised when a normalized candle violates an OHLCV invariant."""


def validate_candle(candle: OHLCVCandle) -> OHLCVCandle:
    """Validate one candle and return it unchanged when it is valid.

    The data boundary requires timezone-aware timestamps and ``Decimal`` numeric
    values so that downstream research does not inherit provider-specific time
    or floating-point behaviour.
    """

    _require_non_empty_text("symbol", candle.symbol)
    _require_non_empty_text("timeframe", candle.timeframe)
    _require_aware_datetime("open_time", candle.open_time)
    _require_aware_datetime("close_time", candle.close_time)

    if candle.open_time >= candle.close_time:
        raise CandleValidationError("open_time must be earlier than close_time")

    for field_name, value in _numeric_fields(candle):
        _require_decimal(field_name, value)

    if candle.volume < Decimal(0):
        raise CandleValidationError("volume must be greater than or equal to zero")

    if min(candle.open, candle.close) < candle.low:
        raise CandleValidationError("low cannot exceed the open or close price")

    if max(candle.open, candle.close) > candle.high:
        raise CandleValidationError("high cannot be lower than the open or close price")

    if candle.low > candle.high:
        raise CandleValidationError("low cannot be greater than high")

    return candle


def validate_candles(candles: Iterable[OHLCVCandle]) -> tuple[OHLCVCandle, ...]:
    """Validate a candle collection and enforce chronological consistency."""

    validated_candles = tuple(validate_candle(candle) for candle in candles)

    for previous, current in pairwise(validated_candles):
        if previous.symbol != current.symbol:
            raise CandleValidationError("all candles must have the same symbol")
        if previous.timeframe != current.timeframe:
            raise CandleValidationError("all candles must have the same timeframe")
        if current.open_time <= previous.open_time:
            raise CandleValidationError("candles must be ordered by open_time")
        if current.open_time < previous.close_time:
            raise CandleValidationError("candles must not overlap")

    return validated_candles


def _numeric_fields(candle: OHLCVCandle) -> tuple[tuple[str, Decimal], ...]:
    return (
        ("open", candle.open),
        ("high", candle.high),
        ("low", candle.low),
        ("close", candle.close),
        ("volume", candle.volume),
    )


def _require_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CandleValidationError(f"{field_name} must be a non-empty string")


def _require_aware_datetime(field_name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CandleValidationError(f"{field_name} must be timezone-aware")


def _require_decimal(field_name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal):
        raise CandleValidationError(f"{field_name} must be a Decimal")
