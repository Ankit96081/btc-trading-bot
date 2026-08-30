"""Tests for the provider-neutral historical CSV downloader."""

import csv
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.data.historical_downloader import (
    ExistingOutputError,
    HistoricalDataDownloader,
    HistoricalDataDownloadError,
    HistoricalDataIntegrityError,
)
from src.data.models import CandleRequest, OHLCVCandle

_START = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)


def make_candle(index: int, **changes: object) -> OHLCVCandle:
    open_time = _START + timedelta(minutes=index)
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "open_time": open_time,
        "close_time": open_time + timedelta(milliseconds=59_999),
        "open": Decimal("65000.10"),
        "high": Decimal("65025.00"),
        "low": Decimal("64990.50"),
        "close": Decimal("65020.75"),
        "volume": Decimal("1.250"),
    }
    values.update(changes)
    return OHLCVCandle(**values)  # type: ignore[arg-type]


class FakeMarketDataProvider:
    """Deterministic in-memory provider; it never makes a network request."""

    def __init__(
        self,
        candles: list[OHLCVCandle],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._candles = candles
        self._fail_on_call = fail_on_call
        self.requests: list[CandleRequest] = []

    def get_candles(self, request: CandleRequest) -> tuple[OHLCVCandle, ...]:
        self.requests.append(request)
        if self._fail_on_call == len(self.requests):
            raise RuntimeError("simulated provider interruption")

        matching = [
            candle
            for candle in self._candles
            if request.start_time is not None
            and request.end_time is not None
            and candle.open_time >= request.start_time
            and candle.open_time <= request.end_time
        ]
        return tuple(matching[: request.limit])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def test_download_writes_a_valid_utc_csv(tmp_path: Path) -> None:
    provider = FakeMarketDataProvider([make_candle(index) for index in range(3)])
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"

    result = HistoricalDataDownloader(provider).download(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=_START,
        end_time=_START + timedelta(minutes=3),
        output_path=output_path,
    )

    assert result.output_path == output_path
    assert result.candle_count == 3
    assert not result.resumed
    assert not output_path.with_name(f"{output_path.name}.partial").exists()
    assert [request.limit for request in provider.requests] == [1_000]

    rows = read_csv_rows(output_path)
    assert list(rows[0]) == [
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_timestamp",
    ]
    assert [row["timestamp"] for row in rows] == [
        "2026-08-30T09:00:00.000Z",
        "2026-08-30T09:01:00.000Z",
        "2026-08-30T09:02:00.000Z",
    ]
    assert rows[0]["symbol"] == "BTC/USDT"
    assert rows[0]["volume"] == "1.250"


def test_download_uses_multiple_batches_and_maintains_chronological_order(
    tmp_path: Path,
) -> None:
    provider = FakeMarketDataProvider([make_candle(index) for index in range(5)])
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"

    HistoricalDataDownloader(provider, batch_size=2).download(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=_START,
        end_time=_START + timedelta(minutes=5),
        output_path=output_path,
    )

    assert [request.start_time for request in provider.requests] == [
        _START,
        _START + timedelta(minutes=2),
        _START + timedelta(minutes=4),
    ]
    assert [row["timestamp"] for row in read_csv_rows(output_path)] == [
        f"2026-08-30T09:0{index}:00.000Z" for index in range(5)
    ]


def test_download_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    provider = FakeMarketDataProvider([make_candle(0), make_candle(1), make_candle(1)])

    with pytest.raises(HistoricalDataIntegrityError, match="duplicate"):
        HistoricalDataDownloader(provider).download(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=_START,
            end_time=_START + timedelta(minutes=3),
            output_path=tmp_path / "data" / "raw" / "btc_usdt_1m.csv",
        )


def test_download_rejects_missing_candles(tmp_path: Path) -> None:
    provider = FakeMarketDataProvider([make_candle(0), make_candle(2)])

    with pytest.raises(HistoricalDataIntegrityError, match="missing candle"):
        HistoricalDataDownloader(provider).download(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=_START,
            end_time=_START + timedelta(minutes=3),
            output_path=tmp_path / "data" / "raw" / "btc_usdt_1m.csv",
        )


def test_download_rejects_invalid_candles(tmp_path: Path) -> None:
    provider = FakeMarketDataProvider([make_candle(0, high=Decimal(64999))])

    with pytest.raises(HistoricalDataIntegrityError, match="Invalid candle"):
        HistoricalDataDownloader(provider).download(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=_START,
            end_time=_START + timedelta(minutes=1),
            output_path=tmp_path / "data" / "raw" / "btc_usdt_1m.csv",
        )


def test_download_resumes_an_interrupted_partial_file(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"
    candles = [make_candle(index) for index in range(5)]
    interrupted_provider = FakeMarketDataProvider(candles, fail_on_call=2)

    with pytest.raises(RuntimeError, match="simulated provider interruption"):
        HistoricalDataDownloader(interrupted_provider, batch_size=2).download(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=_START,
            end_time=_START + timedelta(minutes=5),
            output_path=output_path,
        )

    resumed_provider = FakeMarketDataProvider(candles)
    result = HistoricalDataDownloader(resumed_provider, batch_size=2).download(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=_START,
        end_time=_START + timedelta(minutes=5),
        output_path=output_path,
    )

    assert result.resumed
    assert result.candle_count == 5
    assert resumed_provider.requests[0].start_time == _START + timedelta(minutes=2)
    assert len(read_csv_rows(output_path)) == 5


def test_download_does_not_overwrite_existing_output_without_instruction(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing data", encoding="utf-8")
    provider = FakeMarketDataProvider([make_candle(0)])

    with pytest.raises(ExistingOutputError, match="overwrite=True"):
        HistoricalDataDownloader(provider).download(
            symbol="BTC/USDT",
            timeframe="1m",
            start_time=_START,
            end_time=_START + timedelta(minutes=1),
            output_path=output_path,
        )

    assert provider.requests == []
    assert output_path.read_text(encoding="utf-8") == "existing data"


def test_download_overwrites_existing_output_only_when_explicitly_requested(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing data", encoding="utf-8")

    HistoricalDataDownloader(FakeMarketDataProvider([make_candle(0)])).download(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=_START,
        end_time=_START + timedelta(minutes=1),
        output_path=output_path,
        overwrite=True,
    )

    assert read_csv_rows(output_path)[0]["timestamp"] == "2026-08-30T09:00:00.000Z"


def test_failed_atomic_publish_leaves_only_resumable_staged_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "data" / "raw" / "btc_usdt_1m.csv"
    candles = [make_candle(index) for index in range(2)]

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk error")))
        with pytest.raises(HistoricalDataDownloadError, match="atomically publish"):
            HistoricalDataDownloader(FakeMarketDataProvider(candles)).download(
                symbol="BTC/USDT",
                timeframe="1m",
                start_time=_START,
                end_time=_START + timedelta(minutes=2),
                output_path=output_path,
            )

    partial_path = output_path.with_name(f"{output_path.name}.partial")
    assert not output_path.exists()
    assert partial_path.exists()

    resuming_provider = FakeMarketDataProvider(candles)
    result = HistoricalDataDownloader(resuming_provider).download(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=_START,
        end_time=_START + timedelta(minutes=2),
        output_path=output_path,
    )

    assert result.resumed
    assert resuming_provider.requests == []
    assert output_path.exists()
    assert not partial_path.exists()
