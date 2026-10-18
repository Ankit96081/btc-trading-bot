"""Tests for the Binance Spot public-market-data adapter."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from src.data.models import CandleRequest
from src.data.providers.binance_spot import (
    BinanceApiError,
    BinanceHttpError,
    BinanceResponseError,
    BinanceSpotMarketDataProvider,
)
from src.data.validation import CandleValidationError


def to_milliseconds(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return (value - epoch) // timedelta(milliseconds=1)


def make_kline(
    open_time: datetime,
    *,
    high: str = "65025.00",
) -> list[int | str]:
    open_time_ms = to_milliseconds(open_time)
    return [
        open_time_ms,
        "65000.10",
        high,
        "64990.50",
        "65020.75",
        "1.250",
        open_time_ms + 59_999,
        "81268.99",
        321,
        "0.600",
        "39000.00",
        "0",
    ]


def response_for(*klines: list[int | str]) -> bytes:
    return json.dumps(list(klines)).encode("utf-8")


def test_provider_converts_a_successful_kline_response() -> None:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    transport = Mock(return_value=response_for(make_kline(open_time)))
    provider = BinanceSpotMarketDataProvider(transport=transport)

    candles = provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))

    assert len(candles) == 1
    candle = candles[0]
    assert candle.symbol == "BTC/USDT"
    assert candle.timeframe == "1m"
    assert candle.open_time == open_time
    assert candle.close_time == open_time + timedelta(milliseconds=59_999)
    assert candle.open == Decimal("65000.10")
    assert candle.high == Decimal("65025.00")
    assert candle.low == Decimal("64990.50")
    assert candle.close == Decimal("65020.75")
    assert candle.volume == Decimal("1.250")

    query = parse_qs(urlparse(transport.call_args.args[0]).query)
    assert query["symbol"] == ["BTCUSDT"]
    assert query["interval"] == ["1m"]
    assert query["limit"] == ["1"]


def test_provider_raises_for_a_binance_api_error_payload() -> None:
    transport = Mock(return_value=b'{"code": -1121, "msg": "Invalid symbol."}')
    provider = BinanceSpotMarketDataProvider(transport=transport)

    with pytest.raises(BinanceApiError, match="Invalid symbol"):
        provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))


def test_provider_raises_for_a_malformed_kline_response() -> None:
    transport = Mock(return_value=response_for([1, "2"]))
    provider = BinanceSpotMarketDataProvider(transport=transport)

    with pytest.raises(BinanceResponseError, match="malformed"):
        provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))


def test_provider_raises_for_an_http_failure() -> None:
    error = HTTPError("https://api.binance.com/api/v3/klines", 503, "Unavailable", None, None)

    sleep = Mock()

    with patch("src.data.providers.binance_spot.urlopen", side_effect=error) as urlopen:
        provider = BinanceSpotMarketDataProvider(sleep=sleep)

        with pytest.raises(BinanceHttpError, match="HTTP 503"):
            provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))

    assert urlopen.call_count == 5
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0, 4.0, 8.0]


def test_provider_retries_transient_failures_with_backoff() -> None:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    transport = Mock(
        side_effect=[
            BinanceHttpError("Binance returned HTTP 429", status=429),
            OSError("connection reset"),
            response_for(make_kline(open_time)),
        ]
    )
    sleep = Mock()
    provider = BinanceSpotMarketDataProvider(transport=transport, sleep=sleep)

    candles = provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))

    assert [candle.open_time for candle in candles] == [open_time]
    assert transport.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 418])
def test_provider_does_not_retry_permanent_http_errors(status: int) -> None:
    transport = Mock(side_effect=BinanceHttpError(f"Binance returned HTTP {status}", status=status))
    sleep = Mock()
    provider = BinanceSpotMarketDataProvider(transport=transport, sleep=sleep)

    with pytest.raises(BinanceHttpError, match=f"HTTP {status}"):
        provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))

    assert transport.call_count == 1
    sleep.assert_not_called()


def test_provider_gives_up_after_max_retries() -> None:
    transport = Mock(side_effect=BinanceHttpError("Binance returned HTTP 429", status=429))
    sleep = Mock()
    provider = BinanceSpotMarketDataProvider(
        transport=transport, max_retries=2, retry_delay_seconds=0.5, sleep=sleep
    )

    with pytest.raises(BinanceHttpError, match="HTTP 429"):
        provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))

    assert transport.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


def test_provider_paginates_when_the_requested_limit_exceeds_binance_maximum() -> None:
    start_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    first_page = [make_kline(start_time + timedelta(minutes=index)) for index in range(1_000)]
    second_page = [
        make_kline(start_time + timedelta(minutes=index)) for index in range(1_000, 1_500)
    ]
    response_bodies = iter((response_for(*first_page), response_for(*second_page)))
    transport: Callable[[str, float], bytes] = Mock(side_effect=lambda *_: next(response_bodies))
    provider = BinanceSpotMarketDataProvider(transport=transport)

    candles = provider.get_candles(
        CandleRequest("BTC/USDT", "1m", start_time=start_time, limit=1_500)
    )

    assert len(candles) == 1_500
    assert candles[0].open_time == start_time
    assert candles[-1].open_time == start_time + timedelta(minutes=1_499)

    first_query = parse_qs(urlparse(transport.call_args_list[0].args[0]).query)
    second_query = parse_qs(urlparse(transport.call_args_list[1].args[0]).query)
    assert first_query["limit"] == ["1000"]
    assert second_query["limit"] == ["500"]
    assert second_query["startTime"] == [str(first_page[-1][0] + 1)]


def test_provider_validates_converted_candles() -> None:
    open_time = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
    transport = Mock(return_value=response_for(make_kline(open_time, high="64999")))
    provider = BinanceSpotMarketDataProvider(transport=transport)

    with pytest.raises(CandleValidationError, match="high cannot be lower"):
        provider.get_candles(CandleRequest("BTC/USDT", "1m", limit=1))
