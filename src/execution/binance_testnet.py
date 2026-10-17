"""Signed Binance Spot *Testnet* client and a broker that trades through it.

Safety: the client refuses every host except the Spot Testnet, so this module
cannot place orders with real funds.  Moving to production requires a
deliberate code change after the go-live checklist (docs/go_live_checklist.md)
is complete.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from src.data.models import OHLCVCandle

from .models import Fill, Order, OrderReason, Side

LOGGER = logging.getLogger(__name__)

TESTNET_BASE_URL = "https://testnet.binance.vision"
_ALLOWED_HOSTS = frozenset({"testnet.binance.vision"})
API_KEY_ENV = "BINANCE_TESTNET_API_KEY"
API_SECRET_ENV = "BINANCE_TESTNET_API_SECRET"
_UNKNOWN_ORDER_CODE = -2013

# (method, url, headers, body) -> (status, response bytes)
SignedTransport = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]


class BinanceTradingError(RuntimeError):
    """Raised when the testnet rejects a request or cannot be reached."""


class BinanceOrderUnknownError(BinanceTradingError):
    """Raised when it is unknown whether an order was placed (e.g. a timeout)."""


@dataclass(frozen=True, slots=True)
class SymbolRules:
    """Exchange filters that every order must satisfy."""

    base_asset: str
    quote_asset: str
    step_size: Decimal
    min_quantity: Decimal
    min_notional: Decimal

    def round_quantity(self, quantity: Decimal) -> Decimal:
        if quantity <= 0:
            return Decimal(0)
        steps = (quantity / self.step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * self.step_size


@dataclass(frozen=True, slots=True)
class Credentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_environment(cls) -> "Credentials":
        key, secret = os.environ.get(API_KEY_ENV), os.environ.get(API_SECRET_ENV)
        if not key or not secret:
            raise BinanceTradingError(
                f"set {API_KEY_ENV} and {API_SECRET_ENV} (create keys at {TESTNET_BASE_URL})"
            )
        return cls(key, secret)


def sign(secret: str, query: str) -> str:
    """HMAC-SHA256 signature of a query string, as Binance requires."""

    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinanceTestnetClient:
    """Minimal signed REST client for the Spot Testnet."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        base_url: str = TESTNET_BASE_URL,
        transport: SignedTransport | None = None,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        recv_window_ms: int = 5_000,
        timeout_seconds: float = 10.0,
    ) -> None:
        host = urlparse(base_url).hostname
        if host not in _ALLOWED_HOSTS:
            raise BinanceTradingError(
                f"refusing to trade on {host!r}: only the Spot Testnet is allowed"
            )
        self._credentials = credentials
        self._base_url = base_url.rstrip("/")
        self._transport = transport or _make_default_transport(timeout_seconds)
        self._clock_ms = clock_ms
        self._recv_window_ms = recv_window_ms
        self._time_offset_ms = 0

    def sync_time(self) -> None:
        """Correct for local clock drift, which would otherwise reject signed calls."""

        server = self._request("GET", "/api/v3/time", {}, signed=False)["serverTime"]
        self._time_offset_ms = int(server) - self._clock_ms()

    def symbol_rules(self, symbol: str) -> SymbolRules:
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)
        (entry,) = info["symbols"]
        filters = {item["filterType"]: item for item in entry["filters"]}
        lot = filters["LOT_SIZE"]
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {"minNotional": "0"}
        return SymbolRules(
            base_asset=entry["baseAsset"],
            quote_asset=entry["quoteAsset"],
            step_size=Decimal(lot["stepSize"]).normalize(),
            min_quantity=Decimal(lot["minQty"]),
            min_notional=Decimal(notional["minNotional"]),
        )

    def balances(self) -> dict[str, Decimal]:
        """Total (free + locked) balance per asset."""

        account = self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"})
        return {
            item["asset"]: Decimal(item["free"]) + Decimal(item["locked"])
            for item in account["balances"]
        }

    def market_order(
        self,
        symbol: str,
        side: Side,
        client_order_id: str,
        *,
        quantity: Decimal | None = None,
        quote_quantity: Decimal | None = None,
    ) -> dict[str, Any]:
        """Place a market order for a base ``quantity`` or a quote amount to spend."""

        if (quantity is None) == (quote_quantity is None):
            raise ValueError("give exactly one of quantity or quote_quantity")
        parameters = {
            "symbol": symbol,
            "side": side.value,
            "type": "MARKET",
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL",
        }
        if quantity is not None:
            parameters["quantity"] = format(quantity, "f")
        else:
            parameters["quoteOrderQty"] = format(quote_quantity, "f")
        try:
            return self._request("POST", "/api/v3/order", parameters)
        except BinanceOrderUnknownError:
            # The request may or may not have reached the exchange.  Look the
            # order up by its ID instead of sending a second one.
            found = self.find_order(symbol, client_order_id)
            if found is None:
                raise
            return found

    def find_order(self, symbol: str, client_order_id: str) -> dict[str, Any] | None:
        try:
            return self._request(
                "GET", "/api/v3/order", {"symbol": symbol, "origClientOrderId": client_order_id}
            )
        except BinanceTradingError as error:
            if f"code {_UNKNOWN_ORDER_CODE}" in str(error):
                return None
            raise

    def my_trades(self, symbol: str, order_id: int) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v3/myTrades", {"symbol": symbol, "orderId": order_id})

    def _request(
        self, method: str, path: str, parameters: Mapping[str, object], *, signed: bool = True
    ) -> Any:
        query_parameters = dict(parameters)
        headers: dict[str, str] = {}
        if signed:
            query_parameters["recvWindow"] = self._recv_window_ms
            query_parameters["timestamp"] = self._clock_ms() + self._time_offset_ms
            query = urlencode(query_parameters)
            query = f"{query}&signature={sign(self._credentials.api_secret, query)}"
            headers["X-MBX-APIKEY"] = self._credentials.api_key
        else:
            query = urlencode(query_parameters)

        url = f"{self._base_url}{path}"
        body: bytes | None = None
        if method == "POST":
            body = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif query:
            url = f"{url}?{query}"

        status, payload = self._transport(method, url, headers, body)
        try:
            decoded = json.loads(payload.decode("utf-8")) if payload else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BinanceTradingError(f"invalid JSON from {path} (HTTP {status})") from error
        if status >= 500:
            raise BinanceOrderUnknownError(f"HTTP {status} from {path}; outcome unknown")
        if status != 200:
            code = decoded.get("code") if isinstance(decoded, dict) else None
            message = decoded.get("msg") if isinstance(decoded, dict) else payload[:200]
            raise BinanceTradingError(f"{path} failed (HTTP {status}, code {code}): {message}")
        return decoded


def _make_default_transport(timeout_seconds: float) -> SignedTransport:
    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        request = Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()
        except (URLError, TimeoutError) as error:
            raise BinanceOrderUnknownError(f"network error: {error}") from error

    return transport


class TestnetBroker:
    """Execute engine orders as immediate market orders on the Spot Testnet.

    Differences from the simulated broker, which the paper-vs-backtest
    comparison makes visible:

    * orders fill right after the decision bar closes, at the real market
      price, instead of at the next bar's modelled open;
    * stops are checked on closed bars and exit at market, so a fast move
      can exit below the stop price;
    * fees are the exchange's actual commissions.
    """

    __test__ = False  # not a pytest test class, despite the name

    def __init__(
        self,
        client: BinanceTestnetClient,
        symbol: str,
        *,
        run_tag: str,
        rules: SymbolRules | None = None,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.exchange_symbol = symbol.replace("/", "").upper()
        self.rules = rules or client.symbol_rules(self.exchange_symbol)
        self.run_tag = "".join(char for char in run_tag if char.isalnum())[:16] or "run"
        self.baseline_base_balance: Decimal | None = None
        self.in_flight_order_id: str | None = None
        self.on_state_change: Callable[[], None] = lambda: None

    def on_bar(self, bar: OHLCVCandle) -> list[Fill]:
        return []

    def submit(self, order: Order, bar: OHLCVCandle) -> list[Fill]:
        return self._market(order.side, order.quantity, order.reason, bar, order.quote_budget)

    def stop_out(self, bar: OHLCVCandle, stop_price: Decimal, quantity: Decimal) -> list[Fill]:
        return self._market(Side.SELL, quantity, OrderReason.STOP_LOSS, bar)

    def close_at(self, bar: OHLCVCandle, quantity: Decimal, reason: OrderReason) -> list[Fill]:
        return self._market(Side.SELL, quantity, reason, bar)

    def snapshot(self) -> dict[str, str | None]:
        return {
            "baseline_base_balance": None
            if self.baseline_base_balance is None
            else str(self.baseline_base_balance),
            "in_flight_order_id": self.in_flight_order_id,
        }

    def restore(self, snapshot: Mapping[str, str | None]) -> None:
        baseline = snapshot.get("baseline_base_balance")
        self.baseline_base_balance = None if baseline is None else Decimal(baseline)
        self.in_flight_order_id = snapshot.get("in_flight_order_id")

    def recover_in_flight_order(self) -> list[Fill]:
        """Return fills for an order sent just before a crash, if it executed."""

        client_order_id = self.in_flight_order_id
        if client_order_id is None:
            return []
        self.in_flight_order_id = None
        response = self.client.find_order(self.exchange_symbol, client_order_id)
        if response is None or response.get("status") not in {"FILLED", "PARTIALLY_FILLED"}:
            LOGGER.warning("in-flight order %s did not execute", client_order_id)
            return []
        trades = self.client.my_trades(self.exchange_symbol, int(response["orderId"]))
        fills = [
            {
                "price": trade["price"],
                "qty": trade["qty"],
                "commission": trade["commission"],
                "commissionAsset": trade["commissionAsset"],
            }
            for trade in trades
        ]
        LOGGER.warning("recovered in-flight order %s", client_order_id)
        return [self._to_fill({**response, "fills": fills}, OrderReason.SIGNAL)]

    def _market(
        self,
        side: Side,
        quantity: Decimal,
        reason: OrderReason,
        bar: OHLCVCandle,
        quote_budget: Decimal | None = None,
    ) -> list[Fill]:
        rounded = self.rules.round_quantity(quantity)
        if rounded < self.rules.min_quantity or rounded * bar.close < self.rules.min_notional:
            LOGGER.info("skipping %s %s: below exchange minimums", side.value, quantity)
            return []
        # Buys spend a fixed quote amount so a price move before execution can
        # never overspend the run's cash; Binance takes the commission in BTC.
        quote_amount: Decimal | None = None
        if side is Side.BUY:
            quote_amount = rounded * bar.close
            if quote_budget is not None:
                quote_amount = min(quote_amount, quote_budget)
            quote_amount = quote_amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        client_order_id = f"{self.run_tag}-{int(bar.open_time.timestamp())}-{side.value[0]}"
        self.in_flight_order_id = client_order_id
        self.on_state_change()  # Persist the ID before the request leaves.
        try:
            response = self.client.market_order(
                self.exchange_symbol,
                side,
                client_order_id,
                quantity=None if quote_amount is not None else rounded,
                quote_quantity=quote_amount,
            )
        except BinanceOrderUnknownError:
            LOGGER.exception("order %s outcome unknown; will reconcile on restart", client_order_id)
            raise
        except BinanceTradingError:
            LOGGER.exception("order %s rejected", client_order_id)
            self.in_flight_order_id = None
            return []
        self.in_flight_order_id = None
        if response.get("status") not in {"FILLED", "PARTIALLY_FILLED"}:
            LOGGER.warning("order %s ended with status %s", client_order_id, response.get("status"))
            return []
        return [self._to_fill(response, reason)]

    def _to_fill(self, response: Mapping[str, Any], reason: OrderReason) -> Fill:
        """Convert an order response into one fill in quote-currency terms.

        Binance deducts a BUY's commission from the base asset received, so
        the position grows by ``executed - commission`` while the quote spent
        is ``executed * price``; recording the commission as a quote fee keeps
        the portfolio's cash and quantity equal to the exchange's balances.
        """

        side = Side(response["side"])
        executed = Decimal(response["executedQty"])
        quote = Decimal(response["cummulativeQuoteQty"])
        price = quote / executed
        base_commission = Decimal(0)
        quote_commission = Decimal(0)
        for part in response.get("fills", []):
            asset, amount = part["commissionAsset"], Decimal(part["commission"])
            if asset == self.rules.base_asset:
                base_commission += amount
            elif asset == self.rules.quote_asset:
                quote_commission += amount
            else:
                LOGGER.warning("commission of %s %s is not tracked in the portfolio", amount, asset)

        quantity = executed - base_commission if side is Side.BUY else executed
        fee = quote_commission + base_commission * price
        return Fill(
            timestamp=_from_ms(response.get("transactTime") or response.get("updateTime")),
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            reason=reason,
            client_order_id=response.get("clientOrderId"),
        )


def _from_ms(value: object) -> datetime:
    return datetime.fromtimestamp(int(str(value)) / 1000, tz=UTC)
