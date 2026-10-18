"""Binance Spot public-market-data adapter.

This module uses only Binance's unauthenticated ``/api/v3/klines`` endpoint.
It does not access account, order, futures, or other authenticated endpoints.
"""

import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..interfaces import MarketDataProvider
from ..models import CandleRequest, OHLCVCandle
from ..validation import validate_candles

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
MAX_KLINES_PER_REQUEST = 1_000
_KLINES_PATH = "/api/v3/klines"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

BINANCE_SPOT_KNOWN_GAPS: tuple[tuple[datetime, datetime], ...] = (
    # Spot trading outage on 2023-03-24; Binance published no 1m klines for
    # 12:40-13:59 UTC.  Intervals are [start, end) in UTC.
    (datetime(2023, 3, 24, 12, 40, tzinfo=UTC), datetime(2023, 3, 24, 14, 0, tzinfo=UTC)),
)
"""Intervals in which Binance Spot has no klines, so their absence is not a download error."""

HttpTransport = Callable[[str, float], bytes]


class BinanceMarketDataError(RuntimeError):
    """Base exception for failures specific to the Binance Spot adapter."""


class BinanceRequestError(BinanceMarketDataError):
    """Raised when a generic candle request cannot be sent to Binance."""


class BinanceHttpError(BinanceMarketDataError):
    """Raised when Binance's public HTTP endpoint cannot be reached or succeeds poorly.

    ``status`` is the HTTP status code, or ``None`` when no response arrived
    (connection failure or timeout).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def transient(self) -> bool:
        """Whether retrying the same request later may succeed.

        HTTP 418 (IP ban) is deliberately not transient: retrying extends the ban.
        """

        return self.status is None or self.status in _TRANSIENT_HTTP_STATUSES


class BinanceApiError(BinanceMarketDataError):
    """Raised when Binance returns an error payload from its public API."""


class BinanceResponseError(BinanceMarketDataError):
    """Raised when a successful HTTP response has an unexpected kline shape."""


class BinanceSpotMarketDataProvider(MarketDataProvider):
    """Fetch historical Spot klines and normalize them into application models.

    Binance permits no more than ``MAX_KLINES_PER_REQUEST`` klines per request.
    Requests above that limit are paginated. A request without ``limit`` retrieves
    at most one maximum-sized page; a full historical downloader is intentionally
    outside this adapter's scope.

    Transient HTTP failures (timeouts, connection errors, HTTP 408/429/5xx) are
    retried up to ``max_retries`` times, waiting ``retry_delay_seconds`` and
    doubling the wait after each attempt.  Other failures are raised at once.
    """

    def __init__(
        self,
        base_url: str = BINANCE_SPOT_BASE_URL,
        timeout_seconds: float = 10.0,
        transport: HttpTransport | None = None,
        *,
        max_retries: int = 4,
        retry_delay_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0 or retry_delay_seconds < 0:
            raise ValueError("max_retries and retry_delay_seconds must not be negative")

        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport or _default_transport
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._sleep = sleep

    def get_candles(self, request: CandleRequest) -> Sequence[OHLCVCandle]:
        """Fetch, normalize, and validate candles for ``request``.

        A slash-delimited application symbol such as ``BTC/USDT`` is converted to
        Binance's public-API symbol format, ``BTCUSDT``, only at the HTTP boundary.
        Returned candles retain the application-provided symbol.
        """

        _validate_request(request)
        total_limit = request.limit or MAX_KLINES_PER_REQUEST

        if request.start_time is None:
            candles = self._fetch_backward(request, total_limit)
        else:
            candles = self._fetch_forward(request, total_limit)

        return validate_candles(candles)

    def _fetch_forward(
        self, request: CandleRequest, total_limit: int
    ) -> list[OHLCVCandle]:
        candles: list[OHLCVCandle] = []
        start_time_ms = _datetime_to_milliseconds(request.start_time)
        end_time_ms = _optional_datetime_to_milliseconds(request.end_time)

        while len(candles) < total_limit:
            page_limit = min(MAX_KLINES_PER_REQUEST, total_limit - len(candles))
            page = self._fetch_page(
                request,
                limit=page_limit,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            )
            if not page:
                break

            if _datetime_to_milliseconds(page[0].open_time) < start_time_ms:
                raise BinanceResponseError("Binance returned a candle before startTime")

            candles.extend(page)
            if len(page) < page_limit:
                break

            next_start_time_ms = _datetime_to_milliseconds(page[-1].open_time) + 1
            if next_start_time_ms <= start_time_ms:
                raise BinanceResponseError("Binance pagination did not advance")
            start_time_ms = next_start_time_ms

        return candles

    def _fetch_backward(
        self, request: CandleRequest, total_limit: int
    ) -> list[OHLCVCandle]:
        pages: list[list[OHLCVCandle]] = []
        remaining = total_limit
        end_time_ms = _optional_datetime_to_milliseconds(request.end_time)

        while remaining > 0:
            page_limit = min(MAX_KLINES_PER_REQUEST, remaining)
            page = self._fetch_page(
                request,
                limit=page_limit,
                start_time_ms=None,
                end_time_ms=end_time_ms,
            )
            if not page:
                break

            pages.append(page)
            remaining -= len(page)
            if len(page) < page_limit:
                break

            next_end_time_ms = _datetime_to_milliseconds(page[0].open_time) - 1
            if end_time_ms is not None and next_end_time_ms >= end_time_ms:
                raise BinanceResponseError("Binance pagination did not advance")
            end_time_ms = next_end_time_ms

        return [candle for page in reversed(pages) for candle in page]

    def _fetch_page(
        self,
        request: CandleRequest,
        *,
        limit: int,
        start_time_ms: int | None,
        end_time_ms: int | None,
    ) -> list[OHLCVCandle]:
        parameters: dict[str, str | int] = {
            "symbol": _to_binance_symbol(request.symbol),
            "interval": request.timeframe,
            "limit": limit,
        }
        if start_time_ms is not None:
            parameters["startTime"] = start_time_ms
        if end_time_ms is not None:
            parameters["endTime"] = end_time_ms

        url = f"{self._base_url}{_KLINES_PATH}?{urlencode(parameters)}"
        payload = self._load_with_retries(url)
        return _convert_klines(payload, request)

    def _load_with_retries(self, url: str) -> Any:
        delay = self._retry_delay_seconds
        for attempt in range(self._max_retries + 1):
            try:
                return _load_response(self._transport, url, self._timeout_seconds)
            except BinanceHttpError as error:
                if not error.transient or attempt == self._max_retries:
                    raise
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


def _default_transport(url: str, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise BinanceHttpError(
                    f"Binance returned HTTP {response.status}", status=response.status
                )
            return response.read()
    except HTTPError as error:
        raise BinanceHttpError(f"Binance returned HTTP {error.code}", status=error.code) from error
    except URLError as error:
        raise BinanceHttpError("Unable to reach Binance Spot public API") from error
    except TimeoutError as error:
        raise BinanceHttpError("Binance Spot public API request timed out") from error


def _load_response(transport: HttpTransport, url: str, timeout_seconds: float) -> Any:
    try:
        response_body = transport(url, timeout_seconds)
    except BinanceMarketDataError:
        raise
    except OSError as error:
        raise BinanceHttpError("Unable to reach Binance Spot public API") from error

    if not isinstance(response_body, bytes):
        raise BinanceResponseError("Binance transport returned a non-bytes response")

    try:
        return json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BinanceResponseError("Binance returned invalid JSON") from error


def _convert_klines(payload: Any, request: CandleRequest) -> list[OHLCVCandle]:
    if isinstance(payload, dict):
        if "code" in payload and "msg" in payload:
            raise BinanceApiError(f"Binance API error {payload['code']}: {payload['msg']}")
        raise BinanceResponseError("Binance returned an unexpected JSON object")

    if not isinstance(payload, list):
        raise BinanceResponseError("Binance kline response must be a list")

    return [_convert_kline(row, request, index) for index, row in enumerate(payload)]


def _convert_kline(row: Any, request: CandleRequest, index: int) -> OHLCVCandle:
    if not isinstance(row, list) or len(row) < 7:
        raise BinanceResponseError(f"Binance kline at index {index} is malformed")

    return OHLCVCandle(
        symbol=request.symbol,
        timeframe=request.timeframe,
        open_time=_milliseconds_to_datetime(row[0], "open time", index),
        open=_to_decimal(row[1], "open", index),
        high=_to_decimal(row[2], "high", index),
        low=_to_decimal(row[3], "low", index),
        close=_to_decimal(row[4], "close", index),
        volume=_to_decimal(row[5], "volume", index),
        close_time=_milliseconds_to_datetime(row[6], "close time", index),
    )


def _milliseconds_to_datetime(value: Any, field_name: str, index: int) -> datetime:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise BinanceResponseError(
            f"Binance kline at index {index} has an invalid {field_name} timestamp"
        )

    try:
        milliseconds = int(value)
        return _EPOCH + timedelta(milliseconds=milliseconds)
    except (OverflowError, ValueError) as error:
        raise BinanceResponseError(
            f"Binance kline at index {index} has an invalid {field_name} timestamp"
        ) from error


def _to_decimal(value: Any, field_name: str, index: int) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BinanceResponseError(
            f"Binance kline at index {index} has an invalid {field_name} value"
        )

    try:
        decimal_value = Decimal(str(value))
    except InvalidOperation as error:
        raise BinanceResponseError(
            f"Binance kline at index {index} has an invalid {field_name} value"
        ) from error

    if not decimal_value.is_finite():
        raise BinanceResponseError(
            f"Binance kline at index {index} has a non-finite {field_name} value"
        )

    return decimal_value


def _validate_request(request: CandleRequest) -> None:
    if not isinstance(request.symbol, str) or not request.symbol.strip():
        raise BinanceRequestError("symbol must be a non-empty string")
    if not isinstance(request.timeframe, str) or not request.timeframe.strip():
        raise BinanceRequestError("timeframe must be a non-empty string")

    _validate_optional_datetime("start_time", request.start_time)
    _validate_optional_datetime("end_time", request.end_time)

    if (
        request.start_time is not None
        and request.end_time is not None
        and request.start_time >= request.end_time
    ):
        raise BinanceRequestError("start_time must be earlier than end_time")

    if request.limit is not None:
        if isinstance(request.limit, bool) or not isinstance(request.limit, int):
            raise BinanceRequestError("limit must be an integer")
        if request.limit <= 0:
            raise BinanceRequestError("limit must be greater than zero")


def _validate_optional_datetime(field_name: str, value: datetime | None) -> None:
    if value is None:
        return
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise BinanceRequestError(f"{field_name} must be timezone-aware")


def _datetime_to_milliseconds(value: datetime | None) -> int:
    if value is None:
        raise BinanceRequestError("start_time must be provided for forward pagination")
    return (value.astimezone(UTC) - _EPOCH) // timedelta(milliseconds=1)


def _optional_datetime_to_milliseconds(value: datetime | None) -> int | None:
    if value is None:
        return None
    return _datetime_to_milliseconds(value)


def _to_binance_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()
