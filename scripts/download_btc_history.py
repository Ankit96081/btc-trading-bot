"""Download a UTC BTC/USDT one-minute historical dataset for research.

Example:
    python scripts/download_btc_history.py \
        --start 2024-01-01T00:00:00Z \
        --end 2024-01-02T00:00:00Z \
        --output data/raw/btcusdt_1m_test.csv

This script uses only Binance's public market-data endpoint through the
project's provider and downloader abstractions.  It uses no credentials.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.historical_downloader import (
    HistoricalDataDownloader,
    HistoricalDataDownloadError,
)
from src.data.providers.binance_spot import (
    BINANCE_SPOT_KNOWN_GAPS,
    BinanceMarketDataError,
    BinanceSpotMarketDataProvider,
)
from src.data.validation import CandleValidationError

_SYMBOL = "BTC/USDT"
_TIMEFRAME = "1m"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the historical BTC download."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        required=True,
        type=_parse_utc_timestamp,
        metavar="UTC_TIMESTAMP",
        help="Inclusive UTC start time, for example 2024-01-01T00:00:00Z.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=_parse_utc_timestamp,
        metavar="UTC_TIMESTAMP",
        help="Exclusive UTC end time, for example 2024-01-02T00:00:00Z.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=_parse_output_path,
        metavar="PATH",
        help="CSV output path under data/raw/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing completed output CSV.",
    )
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments without contacting Binance."""

    parser = build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.start >= parsed.end:
        parser.error("--start must be earlier than --end")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the requested download and return a process-compatible exit status."""

    parsed = parse_arguments(arguments)
    print(f"Requested start: {_format_utc(parsed.start)}")
    print(f"Requested end: {_format_utc(parsed.end)}")
    print(f"Output path: {parsed.output}")

    try:
        provider = BinanceSpotMarketDataProvider()
        downloader = HistoricalDataDownloader(provider, known_gaps=BINANCE_SPOT_KNOWN_GAPS)
        result = downloader.download(
            symbol=_SYMBOL,
            timeframe=_TIMEFRAME,
            start_time=parsed.start,
            end_time=parsed.end,
            output_path=parsed.output,
            overwrite=parsed.overwrite,
        )
    except (
        BinanceMarketDataError,
        CandleValidationError,
        HistoricalDataDownloadError,
        OSError,
    ) as error:
        print(f"Historical download failed: {error}", file=sys.stderr)
        return 1

    print(f"Candles downloaded: {result.candle_count}")
    print(f"Dataset saved to: {result.output_path}")
    return 0


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a valid ISO-8601 timestamp") from error

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("must include a UTC offset; naive timestamps are not allowed")
    if timestamp.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError("must be expressed in UTC using Z or +00:00")

    normalized = timestamp.astimezone(UTC)
    if normalized.second != 0 or normalized.microsecond != 0:
        raise argparse.ArgumentTypeError("must be aligned to a one-minute boundary")
    return normalized


def _parse_output_path(value: str) -> Path:
    candidate = Path(value)
    output_path = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    raw_data_dir = RAW_DATA_DIR.resolve()

    try:
        output_path.relative_to(raw_data_dir)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be located under data/raw/") from error

    if output_path == raw_data_dir or output_path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError("must be a CSV file path under data/raw/")
    return output_path


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
