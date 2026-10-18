"""Tests for the BTC historical-download command-line script."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import download_btc_history
from src.data.historical_downloader import HistoricalDataDownloadError
from src.data.validation import CandleValidationError


def test_parse_arguments_accepts_utc_range_and_raw_csv_path() -> None:
    parsed = download_btc_history.parse_arguments(
        (
            "--start",
            "2024-01-01T00:00:00Z",
            "--end",
            "2024-01-02T00:00:00+00:00",
            "--output",
            "data/raw/btcusdt_1m_test.csv",
        )
    )

    assert parsed.start == datetime(2024, 1, 1, tzinfo=UTC)
    assert parsed.end == datetime(2024, 1, 2, tzinfo=UTC)
    assert parsed.output == download_btc_history.RAW_DATA_DIR / "btcusdt_1m_test.csv"
    assert not parsed.overwrite


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            (
                "--start",
                "2024-01-01T00:00:00",
                "--end",
                "2024-01-02T00:00:00Z",
                "--output",
                "data/raw/btcusdt_1m_test.csv",
            ),
            "naive timestamps",
        ),
        (
            (
                "--start",
                "2024-01-01T00:00:00+05:30",
                "--end",
                "2024-01-02T00:00:00Z",
                "--output",
                "data/raw/btcusdt_1m_test.csv",
            ),
            "expressed in UTC",
        ),
        (
            (
                "--start",
                "2024-01-01T00:00:00Z",
                "--end",
                "2024-01-01T00:00:00Z",
                "--output",
                "data/raw/btcusdt_1m_test.csv",
            ),
            "earlier than",
        ),
        (
            (
                "--start",
                "2024-01-01T00:00:00Z",
                "--end",
                "2024-01-02T00:00:00Z",
                "--output",
                "data/processed/btcusdt_1m_test.csv",
            ),
            "under data/raw",
        ),
    ],
)
def test_parse_arguments_rejects_invalid_requests(
    arguments: tuple[str, ...], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        download_btc_history.parse_arguments(arguments)

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_main_uses_provider_and_downloader_without_a_network_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = Mock()
    provider_factory = Mock(return_value=provider)
    downloader = Mock()
    downloader_factory = Mock(return_value=downloader)
    output_path = download_btc_history.RAW_DATA_DIR / "btcusdt_1m_test.csv"
    downloader.download.return_value = SimpleNamespace(candle_count=1_440, output_path=output_path)
    monkeypatch.setattr(download_btc_history, "BinanceSpotMarketDataProvider", provider_factory)
    monkeypatch.setattr(download_btc_history, "HistoricalDataDownloader", downloader_factory)

    exit_code = download_btc_history.main(
        (
            "--start",
            "2024-01-01T00:00:00Z",
            "--end",
            "2024-01-02T00:00:00Z",
            "--output",
            "data/raw/btcusdt_1m_test.csv",
        )
    )

    assert exit_code == 0
    provider_factory.assert_called_once_with()
    downloader_factory.assert_called_once_with(
        provider, known_gaps=download_btc_history.BINANCE_SPOT_KNOWN_GAPS
    )
    downloader.download.assert_called_once_with(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 1, 2, tzinfo=UTC),
        output_path=output_path,
        overwrite=False,
    )
    output = capsys.readouterr().out
    assert "Requested start: 2024-01-01T00:00:00Z" in output
    assert "Candles downloaded: 1440" in output


def test_main_returns_nonzero_when_download_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloader = Mock()
    downloader.download.side_effect = HistoricalDataDownloadError("simulated download failure")
    monkeypatch.setattr(download_btc_history, "BinanceSpotMarketDataProvider", Mock())
    monkeypatch.setattr(download_btc_history, "HistoricalDataDownloader", Mock(return_value=downloader))

    exit_code = download_btc_history.main(
        (
            "--start",
            "2024-01-01T00:00:00Z",
            "--end",
            "2024-01-02T00:00:00Z",
            "--output",
            "data/raw/btcusdt_1m_test.csv",
            "--overwrite",
        )
    )

    assert exit_code == 1
    assert "Historical download failed: simulated download failure" in capsys.readouterr().err


def test_main_returns_nonzero_when_candle_validation_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloader = Mock()
    downloader.download.side_effect = CandleValidationError("simulated invalid candle")
    monkeypatch.setattr(download_btc_history, "BinanceSpotMarketDataProvider", Mock())
    monkeypatch.setattr(download_btc_history, "HistoricalDataDownloader", Mock(return_value=downloader))

    exit_code = download_btc_history.main(
        (
            "--start",
            "2024-01-01T00:00:00Z",
            "--end",
            "2024-01-02T00:00:00Z",
            "--output",
            "data/raw/btcusdt_1m_test.csv",
        )
    )

    assert exit_code == 1
    assert "Historical download failed: simulated invalid candle" in capsys.readouterr().err
