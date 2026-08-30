"""Provider-neutral, resumable CSV downloads for one-minute historical candles.

The downloader deliberately knows nothing about HTTP or Binance endpoints.  It
asks a :class:`MarketDataProvider` for bounded candle batches, validates each
batch, and publishes the completed CSV only after validating the whole staged
dataset.  An interrupted download remains in a ``.partial`` sibling file that
can be safely resumed by a later invocation with the same request.
"""

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .interfaces import MarketDataProvider
from .models import CandleRequest, OHLCVCandle
from .validation import CandleValidationError, validate_candle

_ONE_MINUTE = timedelta(minutes=1)
_ONE_MILLISECOND = timedelta(milliseconds=1)
_CSV_COLUMNS = (
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


class HistoricalDataDownloadError(RuntimeError):
    """Base error for historical-data download failures."""


class ExistingOutputError(HistoricalDataDownloadError):
    """Raised when the target CSV already exists and overwrite was not requested."""


class HistoricalDataIntegrityError(HistoricalDataDownloadError):
    """Raised when a provider response or staged CSV is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class HistoricalDownloadResult:
    """Details of a completed historical-data download."""

    output_path: Path
    candle_count: int
    resumed: bool


@dataclass(frozen=True, slots=True)
class _DownloadProgress:
    """Validated staged-data progress used to resume a request."""

    candle_count: int
    next_open_time: datetime


class HistoricalDataDownloader:
    """Download and atomically publish one-minute OHLCV history from a provider.

    The requested interval is ``[start_time, end_time)`` in UTC.  Restricting
    this first downloader to ``1m`` makes missing-timestamp detection exact and
    keeps the implementation aligned with the project's current BTC/USDT
    research scope.
    """

    def __init__(self, provider: MarketDataProvider, *, batch_size: int = 1_000) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._provider = provider
        self._batch_size = batch_size

    def download(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        output_path: str | Path,
        overwrite: bool = False,
    ) -> HistoricalDownloadResult:
        """Download, validate, and atomically publish a historical CSV.

        Existing completed output is never overwritten unless ``overwrite`` is
        true.  Before publication, data is written to ``<output>.partial`` in
        the same directory; a later call without ``overwrite`` resumes that
        staged file after revalidating it.
        """

        normalized_symbol = _validate_non_empty_text("symbol", symbol)
        normalized_timeframe = _validate_timeframe(timeframe)
        normalized_start = _validate_boundary("start_time", start_time)
        normalized_end = _validate_boundary("end_time", end_time)
        if normalized_start >= normalized_end:
            raise ValueError("start_time must be earlier than end_time")

        target_path = Path(output_path)
        progress, resumed = self._prepare_output(
            target_path,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            start_time=normalized_start,
            end_time=normalized_end,
            overwrite=overwrite,
        )

        partial_path = _partial_path(target_path)
        provider_end_time = normalized_end - _ONE_MILLISECOND
        next_open_time = progress.next_open_time

        while next_open_time < normalized_end:
            request = CandleRequest(
                symbol=normalized_symbol,
                timeframe=normalized_timeframe,
                start_time=next_open_time,
                end_time=provider_end_time,
                limit=self._batch_size,
            )
            candles = self._provider.get_candles(request)
            next_open_time = _validate_provider_batch(
                candles,
                expected_open_time=next_open_time,
                end_time=normalized_end,
                symbol=normalized_symbol,
                timeframe=normalized_timeframe,
                batch_size=self._batch_size,
            )
            _append_candles(partial_path, candles)

        completed_progress = _inspect_staged_csv(
            partial_path,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            start_time=normalized_start,
            end_time=normalized_end,
        )
        if completed_progress.next_open_time != normalized_end:
            raise HistoricalDataIntegrityError(
                "staged dataset ended before the requested end_time"
            )

        try:
            os.replace(partial_path, target_path)
        except OSError as error:
            raise HistoricalDataDownloadError(
                f"Unable to atomically publish historical data to {target_path}"
            ) from error

        return HistoricalDownloadResult(
            output_path=target_path,
            candle_count=completed_progress.candle_count,
            resumed=resumed,
        )

    def _prepare_output(
        self,
        target_path: Path,
        *,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        overwrite: bool,
    ) -> tuple[_DownloadProgress, bool]:
        partial_path = _partial_path(target_path)
        if target_path.exists():
            if target_path.is_dir():
                raise HistoricalDataDownloadError(
                    f"output_path must name a CSV file, not a directory: {target_path}"
                )
            if not overwrite:
                raise ExistingOutputError(
                    f"Historical data already exists at {target_path}; pass overwrite=True "
                    "to replace it"
                )

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite and partial_path.exists():
            if partial_path.is_dir():
                raise HistoricalDataDownloadError(
                    f"staged output path is a directory: {partial_path}"
                )
            try:
                partial_path.unlink()
            except OSError as error:
                raise HistoricalDataDownloadError(
                    f"Unable to discard staged historical data at {partial_path}"
                ) from error

        if partial_path.exists():
            return (
                _inspect_staged_csv(
                    partial_path,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                ),
                True,
            )

        _create_staged_csv(partial_path)
        return _DownloadProgress(candle_count=0, next_open_time=start_time), False


def _validate_provider_batch(
    candles: object,
    *,
    expected_open_time: datetime,
    end_time: datetime,
    symbol: str,
    timeframe: str,
    batch_size: int,
) -> datetime:
    if not isinstance(candles, Sequence):
        raise HistoricalDataIntegrityError("provider returned a non-sequence candle batch")
    if not candles:
        raise HistoricalDataIntegrityError(
            f"provider returned no candle for expected timestamp {_format_timestamp(expected_open_time)}"
        )
    if len(candles) > batch_size:
        raise HistoricalDataIntegrityError("provider returned more candles than requested")

    next_open_time = expected_open_time
    for candle in candles:
        if not isinstance(candle, OHLCVCandle):
            raise HistoricalDataIntegrityError("provider returned a non-OHLCVCandle value")
        _validate_candle_for_download(
            candle,
            expected_open_time=next_open_time,
            end_time=end_time,
            symbol=symbol,
            timeframe=timeframe,
        )
        next_open_time += _ONE_MINUTE

    return next_open_time


def _validate_candle_for_download(
    candle: OHLCVCandle,
    *,
    expected_open_time: datetime,
    end_time: datetime,
    symbol: str,
    timeframe: str,
) -> None:
    try:
        validate_candle(candle)
    except CandleValidationError as error:
        raise HistoricalDataIntegrityError(f"Invalid candle: {error}") from error

    if candle.symbol != symbol:
        raise HistoricalDataIntegrityError("provider returned a candle for a different symbol")
    if candle.timeframe != timeframe:
        raise HistoricalDataIntegrityError("provider returned a candle for a different timeframe")

    open_time = candle.open_time.astimezone(UTC)
    if open_time.microsecond % 1_000 != 0:
        raise HistoricalDataIntegrityError("candle timestamps must have millisecond precision")
    if open_time >= end_time:
        raise HistoricalDataIntegrityError("provider returned a candle at or after end_time")
    if open_time < expected_open_time:
        raise HistoricalDataIntegrityError(
            f"duplicate or out-of-order timestamp {_format_timestamp(open_time)}"
        )
    if open_time > expected_open_time:
        raise HistoricalDataIntegrityError(
            f"missing candle timestamp {_format_timestamp(expected_open_time)}"
        )


def _create_staged_csv(partial_path: Path) -> None:
    try:
        with partial_path.open("x", newline="", encoding="utf-8") as file_handle:
            csv.DictWriter(file_handle, fieldnames=_CSV_COLUMNS).writeheader()
            file_handle.flush()
            os.fsync(file_handle.fileno())
    except OSError as error:
        raise HistoricalDataDownloadError(
            f"Unable to create staged historical data at {partial_path}"
        ) from error


def _append_candles(partial_path: Path, candles: Sequence[OHLCVCandle]) -> None:
    try:
        with partial_path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=_CSV_COLUMNS)
            for candle in candles:
                writer.writerow(_candle_to_row(candle))
            file_handle.flush()
            os.fsync(file_handle.fileno())
    except OSError as error:
        raise HistoricalDataDownloadError(
            f"Unable to append staged historical data at {partial_path}"
        ) from error


def _inspect_staged_csv(
    partial_path: Path,
    *,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
) -> _DownloadProgress:
    expected_open_time = start_time
    candle_count = 0

    try:
        with partial_path.open(newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
                raise HistoricalDataIntegrityError("staged CSV has an unexpected schema")

            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise HistoricalDataIntegrityError("staged CSV contains a malformed row")
                candle = _row_to_candle(row)
                _validate_candle_for_download(
                    candle,
                    expected_open_time=expected_open_time,
                    end_time=end_time,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                expected_open_time += _ONE_MINUTE
                candle_count += 1
    except HistoricalDataIntegrityError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error, InvalidOperation, ValueError) as error:
        raise HistoricalDataIntegrityError(
            f"Unable to validate staged historical data at {partial_path}"
        ) from error

    return _DownloadProgress(candle_count=candle_count, next_open_time=expected_open_time)


def _candle_to_row(candle: OHLCVCandle) -> dict[str, str]:
    return {
        "timestamp": _format_timestamp(candle.open_time),
        "symbol": candle.symbol,
        "timeframe": candle.timeframe,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": str(candle.volume),
        "close_timestamp": _format_timestamp(candle.close_time),
    }


def _row_to_candle(row: dict[str | None, str | None]) -> OHLCVCandle:
    return OHLCVCandle(
        symbol=_required_csv_value(row, "symbol"),
        timeframe=_required_csv_value(row, "timeframe"),
        open_time=_parse_timestamp(_required_csv_value(row, "timestamp")),
        close_time=_parse_timestamp(_required_csv_value(row, "close_timestamp")),
        open=_parse_decimal(row, "open"),
        high=_parse_decimal(row, "high"),
        low=_parse_decimal(row, "low"),
        close=_parse_decimal(row, "close"),
        volume=_parse_decimal(row, "volume"),
    )


def _required_csv_value(row: dict[str | None, str | None], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        raise HistoricalDataIntegrityError(f"staged CSV has no {field_name} value")
    return value


def _parse_decimal(row: dict[str | None, str | None], field_name: str) -> Decimal:
    try:
        value = Decimal(_required_csv_value(row, field_name))
    except InvalidOperation as error:
        raise HistoricalDataIntegrityError(
            f"staged CSV has an invalid {field_name} value"
        ) from error
    if not value.is_finite():
        raise HistoricalDataIntegrityError(f"staged CSV has a non-finite {field_name} value")
    return value


def _parse_timestamp(value: str) -> datetime:
    if not value.endswith("Z"):
        raise HistoricalDataIntegrityError("staged CSV timestamps must be UTC and end with Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise HistoricalDataIntegrityError("staged CSV has an invalid timestamp") from error

    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise HistoricalDataIntegrityError("staged CSV timestamps must be UTC")
    normalized = parsed.astimezone(UTC)
    if _format_timestamp(normalized) != value:
        raise HistoricalDataIntegrityError("staged CSV timestamps use an inconsistent format")
    return normalized


def _validate_non_empty_text(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_timeframe(value: str) -> str:
    normalized = _validate_non_empty_text("timeframe", value)
    if normalized != "1m":
        raise ValueError("only the 1m timeframe is supported by this downloader")
    return normalized


def _validate_boundary(field_name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    normalized = value.astimezone(UTC)
    if normalized.second != 0 or normalized.microsecond != 0:
        raise ValueError(f"{field_name} must be aligned to a one-minute boundary")
    return normalized


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _partial_path(target_path: Path) -> Path:
    return target_path.with_name(f"{target_path.name}.partial")
