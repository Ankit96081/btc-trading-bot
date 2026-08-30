"""Manually verify public Binance Spot historical-candle connectivity.

Run from the repository root with:
    python scripts/binance_market_data_smoke.py

This is intentionally a small, opt-in connectivity and data-integrity check.
It uses no credentials and never accesses authenticated Binance endpoints.
"""

import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.historical_downloader import HistoricalDataDownloader
from src.data.providers.binance_spot import BinanceSpotMarketDataProvider

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
START_TIME = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
END_TIME = START_TIME + timedelta(hours=1)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "binance_btc_usdt_1m_20240101T000000Z_20240101T010000Z.csv"
)


def main() -> None:
    """Download one fixed UTC hour and print a concise dataset summary."""

    provider = BinanceSpotMarketDataProvider()
    downloader = HistoricalDataDownloader(provider)
    result = downloader.download(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_time=START_TIME,
        end_time=END_TIME,
        output_path=OUTPUT_PATH,
    )

    rows = _read_rows(result.output_path)
    timestamps_are_utc = _timestamps_are_utc(rows)
    if not timestamps_are_utc:
        raise RuntimeError("Downloaded CSV contains a non-UTC timestamp")

    print(f"Requested start: {_format_utc(START_TIME)}")
    print(f"Requested end: {_format_utc(END_TIME)}")
    print(f"Candles downloaded: {result.candle_count}")
    print(f"First candle timestamp: {rows[0]['timestamp']}")
    print(f"Last candle timestamp: {rows[-1]['timestamp']}")
    print(f"First close: {rows[0]['close']}")
    print(f"Last close: {rows[-1]['close']}")
    print("Timestamps are UTC: yes")
    print(f"CSV saved to: {result.output_path}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise RuntimeError("Downloaded CSV is empty")
    return rows


def _timestamps_are_utc(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for field_name in ("timestamp", "close_timestamp"):
            timestamp = datetime.fromisoformat(row[field_name])
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                return False
    return True


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
