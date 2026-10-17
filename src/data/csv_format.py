"""The project's canonical OHLCV CSV schema.

Both the historical downloader (which writes these files) and the dataset
reader (which loads them for research) use this module so that the on-disk
format has exactly one definition.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

from .models import OHLCVCandle

CANDLE_CSV_COLUMNS = (
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_timestamp",
)


class CandleCsvError(ValueError):
    """Raised when a CSV row does not follow the canonical candle schema."""


def candle_to_csv_row(candle: OHLCVCandle) -> dict[str, str]:
    """Serialize a candle into a canonical CSV row."""

    return {
        "timestamp": format_utc_timestamp(candle.open_time),
        "symbol": candle.symbol,
        "timeframe": candle.timeframe,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": str(candle.volume),
        "close_timestamp": format_utc_timestamp(candle.close_time),
    }


def csv_row_to_candle(row: Mapping[str | None, str | None]) -> OHLCVCandle:
    """Parse a canonical CSV row into a candle.

    The candle is not checked for OHLCV invariants here; callers pass it to
    :func:`src.data.validation.validate_candle` as part of their own checks.
    """

    if None in row or any(value is None for value in row.values()):
        raise CandleCsvError("CSV contains a malformed row")

    return OHLCVCandle(
        symbol=_required_value(row, "symbol"),
        timeframe=_required_value(row, "timeframe"),
        open_time=parse_utc_timestamp(_required_value(row, "timestamp")),
        close_time=parse_utc_timestamp(_required_value(row, "close_timestamp")),
        open=_parse_decimal(row, "open"),
        high=_parse_decimal(row, "high"),
        low=_parse_decimal(row, "low"),
        close=_parse_decimal(row, "close"),
        volume=_parse_decimal(row, "volume"),
    )


def format_utc_timestamp(value: datetime) -> str:
    """Format a timestamp as UTC with millisecond precision and a ``Z`` suffix."""

    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc_timestamp(value: str) -> datetime:
    """Parse a timestamp written by :func:`format_utc_timestamp`."""

    if not value.endswith("Z"):
        raise CandleCsvError("CSV timestamps must be UTC and end with Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CandleCsvError("CSV has an invalid timestamp") from error

    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise CandleCsvError("CSV timestamps must be UTC")
    normalized = parsed.astimezone(UTC)
    if format_utc_timestamp(normalized) != value:
        raise CandleCsvError("CSV timestamps use an inconsistent format")
    return normalized


def _required_value(row: Mapping[str | None, str | None], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        raise CandleCsvError(f"CSV has no {field_name} value")
    return value


def _parse_decimal(row: Mapping[str | None, str | None], field_name: str) -> Decimal:
    try:
        value = Decimal(_required_value(row, field_name))
    except InvalidOperation as error:
        raise CandleCsvError(f"CSV has an invalid {field_name} value") from error
    if not value.is_finite():
        raise CandleCsvError(f"CSV has a non-finite {field_name} value")
    return value
