"""Load published candle CSV files into validated, gap-checked datasets."""

import csv
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

from .csv_format import CANDLE_CSV_COLUMNS, CandleCsvError, csv_row_to_candle
from .models import OHLCVCandle
from .timeframes import timeframe_to_timedelta
from .validation import CandleValidationError, validate_candle


class DatasetError(ValueError):
    """Raised when candle CSV files cannot be combined into a consistent dataset."""


def load_candles(
    paths: Path | str | Iterable[Path | str],
    *,
    allow_gaps: bool = False,
) -> tuple[OHLCVCandle, ...]:
    """Load one or more canonical candle CSV files as a single ordered dataset.

    Files may be given in any order; they are combined by timestamp.  Every
    candle is validated, all candles must share a symbol and timeframe, and
    duplicate or overlapping timestamps are rejected.  Missing bars are
    rejected unless ``allow_gaps`` is true, because silently skipped bars make
    backtests look smoother than reality.
    """

    file_paths = _normalize_paths(paths)
    if not file_paths:
        raise DatasetError("at least one CSV path is required")

    candles: list[OHLCVCandle] = []
    for path in file_paths:
        candles.extend(_read_csv(path))
    if not candles:
        raise DatasetError("the dataset contains no candles")

    candles.sort(key=lambda candle: candle.open_time)
    _check_consistency(candles, allow_gaps=allow_gaps)
    return tuple(candles)


def find_gaps(candles: Iterable[OHLCVCandle]) -> list[tuple[OHLCVCandle, OHLCVCandle]]:
    """Return ``(before, after)`` pairs of consecutive candles with missing bars between them."""

    gaps: list[tuple[OHLCVCandle, OHLCVCandle]] = []
    previous: OHLCVCandle | None = None
    for candle in candles:
        if previous is not None and candle.open_time - previous.open_time > timeframe_to_timedelta(
            candle.timeframe
        ):
            gaps.append((previous, candle))
        previous = candle
    return gaps


def _normalize_paths(paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _read_csv(path: Path) -> list[OHLCVCandle]:
    try:
        with path.open(newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            if tuple(reader.fieldnames or ()) != CANDLE_CSV_COLUMNS:
                raise DatasetError(f"{path} does not use the canonical candle CSV schema")
            return [validate_candle(csv_row_to_candle(row)) for row in reader]
    except (CandleCsvError, CandleValidationError) as error:
        raise DatasetError(f"{path} contains an invalid candle: {error}") from error
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DatasetError(f"Unable to read candle CSV {path}") from error


def _check_consistency(candles: list[OHLCVCandle], *, allow_gaps: bool) -> None:
    symbol = candles[0].symbol
    timeframe = candles[0].timeframe
    try:
        bar_duration = timeframe_to_timedelta(timeframe)
    except ValueError as error:
        raise DatasetError(str(error)) from error

    for previous, current in pairwise(candles):
        if current.symbol != symbol:
            raise DatasetError("all candles must have the same symbol")
        if current.timeframe != timeframe:
            raise DatasetError("all candles must have the same timeframe")
        step = current.open_time - previous.open_time
        if step < bar_duration:
            raise DatasetError(
                f"duplicate or overlapping candle at {current.open_time.isoformat()}"
            )
        if step > bar_duration and not allow_gaps:
            raise DatasetError(
                f"missing candles between {previous.open_time.isoformat()} "
                f"and {current.open_time.isoformat()}"
            )
