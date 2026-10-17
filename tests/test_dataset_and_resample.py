"""Tests for loading candle CSVs, timeframes, and resampling."""

import csv
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.data.csv_format import CANDLE_CSV_COLUMNS, candle_to_csv_row
from src.data.dataset import DatasetError, find_gaps, load_candles
from src.data.models import OHLCVCandle
from src.data.resample import BarAggregator, ResampleError, resample_candles
from src.data.timeframes import (
    bars_per_year,
    timeframe_to_minutes,
    timeframe_to_timedelta,
)
from tests.factories import DEFAULT_START, bars_from_closes, make_bar


def write_csv(path: Path, candles: list[OHLCVCandle]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CANDLE_CSV_COLUMNS)
        writer.writeheader()
        for candle in candles:
            writer.writerow(candle_to_csv_row(candle))
    return path


def minute_bars(count: int, offset: int = 0) -> list[OHLCVCandle]:
    return bars_from_closes(
        [100 + index for index in range(offset, offset + count)],
        timeframe="1m",
        start=DEFAULT_START + timedelta(minutes=offset),
        spread="0.5",
    )


@pytest.mark.parametrize(
    ("timeframe", "minutes"), [("1m", 1), ("15m", 15), ("1h", 60), ("4h", 240), ("1d", 1440)]
)
def test_timeframe_parsing(timeframe: str, minutes: int) -> None:
    assert timeframe_to_minutes(timeframe) == minutes
    assert timeframe_to_timedelta(timeframe) == timedelta(minutes=minutes)


@pytest.mark.parametrize("timeframe", ["", "0m", "1w", "h1", "1.5h"])
def test_timeframe_parsing_rejects_unsupported_values(timeframe: str) -> None:
    with pytest.raises(ValueError):
        timeframe_to_minutes(timeframe)


def test_bars_per_year_uses_a_24_7_calendar() -> None:
    assert bars_per_year("1h") == 8_760
    assert bars_per_year("1d") == 365


def test_load_candles_combines_files_in_time_order(tmp_path: Path) -> None:
    candles = minute_bars(6)
    later = write_csv(tmp_path / "b.csv", candles[3:])
    earlier = write_csv(tmp_path / "a.csv", candles[:3])

    loaded = load_candles([later, earlier])

    assert loaded == tuple(candles)


def test_load_candles_rejects_gaps_unless_allowed(tmp_path: Path) -> None:
    candles = minute_bars(5)
    path = write_csv(tmp_path / "gap.csv", candles[:2] + candles[3:])

    with pytest.raises(DatasetError, match="missing candles"):
        load_candles(path)

    loaded = load_candles(path, allow_gaps=True)
    assert len(loaded) == 4
    assert find_gaps(loaded) == [(candles[1], candles[3])]


def test_load_candles_rejects_duplicates(tmp_path: Path) -> None:
    candles = minute_bars(3)
    first = write_csv(tmp_path / "a.csv", candles)
    second = write_csv(tmp_path / "b.csv", candles[2:])

    with pytest.raises(DatasetError, match="duplicate"):
        load_candles([first, second])


def test_load_candles_rejects_invalid_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "bad.csv", minute_bars(2))
    text = path.read_text(encoding="utf-8").replace("BTC/USDT,1m,100", "BTC/USDT,1m,abc", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DatasetError, match="invalid open value"):
        load_candles(path)


def test_load_candles_rejects_foreign_schema(tmp_path: Path) -> None:
    path = tmp_path / "foreign.csv"
    path.write_text("time,price\n1,2\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="canonical"):
        load_candles(path)


def test_resample_aggregates_complete_bars() -> None:
    candles = minute_bars(10)

    bars = resample_candles(candles, "5m")

    assert len(bars) == 2
    first = bars[0]
    assert first.timeframe == "5m"
    assert first.open_time == DEFAULT_START
    assert first.close_time == DEFAULT_START + timedelta(minutes=5, milliseconds=-1)
    assert first.open == candles[0].open
    assert first.close == candles[4].close
    assert first.high == max(candle.high for candle in candles[:5])
    assert first.low == min(candle.low for candle in candles[:5])
    assert first.volume == Decimal(5)


def test_resample_drops_buckets_with_gaps_and_trailing_partials() -> None:
    candles = minute_bars(12)
    with_gap = candles[:2] + candles[3:]

    bars = resample_candles(with_gap, "5m")

    assert [bar.open_time for bar in bars] == [DEFAULT_START + timedelta(minutes=5)]


def test_resample_aligns_to_the_utc_epoch() -> None:
    candles = minute_bars(8, offset=3)

    bars = resample_candles(candles, "5m")

    assert [bar.open_time for bar in bars] == [DEFAULT_START + timedelta(minutes=5)]


def test_resample_rejects_incompatible_timeframes() -> None:
    with pytest.raises(ResampleError, match="cannot resample"):
        resample_candles(bars_from_closes([1, 2], timeframe="15m"), "5m")
    with pytest.raises(ResampleError, match="cannot resample"):
        resample_candles(bars_from_closes([1, 2], timeframe="15m"), "20m")


def test_bar_aggregator_rejects_out_of_order_candles() -> None:
    aggregator = BarAggregator("5m")
    candles = minute_bars(2)
    aggregator.add(candles[1])

    with pytest.raises(ResampleError, match="increasing"):
        aggregator.add(candles[0])


def test_factory_bars_are_valid() -> None:
    bar = make_bar(DEFAULT_START, open=1, high=2, low="0.5", close="1.5")
    assert bar.high == Decimal(2)
