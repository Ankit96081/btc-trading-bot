"""Tests for the signed testnet client, testnet broker, and reconciliation (no network)."""

import json
from collections.abc import Mapping
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import pytest

from src.core import EngineConfig, Portfolio, TradingEngine
from src.database import TradingStore
from src.execution import Order, Side
from src.execution.binance_testnet import (
    BinanceOrderUnknownError,
    BinanceTestnetClient,
    BinanceTradingError,
    Credentials,
    TestnetBroker,
    sign,
)
from src.paper import ClosedBarFeed, LiveTrader, TraderSettings
from src.paper.reconciliation import reconcile_with_exchange
from src.risk import RiskManager
from src.strategies import BuyAndHold
from tests.factories import bars_from_closes
from tests.test_paper_trading import Clock, FakeExchange, after_close, market

CREDENTIALS = Credentials("test-key", "test-secret")
NOW_MS = 1_700_000_000_000


class FakeTestnet:
    """An in-memory Spot Testnet: checks signatures, fills market orders at ``price``."""

    def __init__(self, price: Decimal = Decimal(100)) -> None:
        self.price = price
        self.balances = {"BTC": Decimal(1), "USDT": Decimal(10000)}
        self.orders: dict[str, dict[str, object]] = {}
        self.trades: dict[int, list[dict[str, str]]] = {}
        self.requests: list[tuple[str, str]] = []
        self.timeout_after_placing = False
        self.fail_lookups = False
        self.fee_rate = Decimal("0.001")

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        parsed = urlparse(url)
        path = parsed.path
        raw = body.decode() if body else parsed.query
        self.requests.append((method, path))
        if path in {"/api/v3/time", "/api/v3/exchangeInfo"}:
            return self._public(path)

        unsigned, _, signature = raw.rpartition("&signature=")
        assert headers["X-MBX-APIKEY"] == CREDENTIALS.api_key
        if signature != sign(CREDENTIALS.api_secret, unsigned):
            return 401, json.dumps({"code": -1022, "msg": "Signature invalid"}).encode()
        params = dict(parse_qsl(unsigned))

        if path == "/api/v3/account":
            balances = [
                {"asset": asset, "free": str(amount), "locked": "0"}
                for asset, amount in self.balances.items()
            ]
            return 200, json.dumps({"balances": balances}).encode()
        if path == "/api/v3/order" and method == "POST":
            return self._place(params)
        if path == "/api/v3/order":
            if self.fail_lookups:
                raise BinanceOrderUnknownError("simulated network failure")
            order = self.orders.get(params["origClientOrderId"])
            if order is None:
                return 400, json.dumps({"code": -2013, "msg": "Order does not exist."}).encode()
            return 200, json.dumps(order).encode()
        if path == "/api/v3/myTrades":
            return 200, json.dumps(self.trades[int(params["orderId"])]).encode()
        raise AssertionError(f"unexpected request {method} {path}")

    def _public(self, path: str) -> tuple[int, bytes]:
        if path == "/api/v3/time":
            return 200, json.dumps({"serverTime": NOW_MS + 1_500}).encode()
        info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00001000",
                         "maxQty": "9000.00000000", "stepSize": "0.00001000"},
                        {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                    ],
                }
            ]
        }
        return 200, json.dumps(info).encode()

    def _place(self, params: dict[str, str]) -> tuple[int, bytes]:
        client_id = params["newClientOrderId"]
        if client_id in self.orders:
            return 400, json.dumps({"code": -2010, "msg": "Duplicate order sent."}).encode()
        side = params["side"]
        if "quoteOrderQty" in params:
            # Like Binance: buy as much as the quote amount allows, in whole lot steps.
            step = Decimal("0.00001")
            affordable = Decimal(params["quoteOrderQty"]) / self.price / step
            quantity = affordable.to_integral_value(rounding="ROUND_DOWN") * step
        else:
            quantity = Decimal(params["quantity"])
        quote = quantity * self.price
        if side == "BUY":
            if quote > self.balances["USDT"]:
                return 400, json.dumps({"code": -2010, "msg": "insufficient balance"}).encode()
            commission, asset = quantity * self.fee_rate, "BTC"
            self.balances["USDT"] -= quote
            self.balances["BTC"] += quantity - commission
        else:
            commission, asset = quote * self.fee_rate, "USDT"
            self.balances["BTC"] -= quantity
            self.balances["USDT"] += quote - commission
        order_id = len(self.orders) + 1
        fill = {"price": str(self.price), "qty": str(quantity),
                "commission": str(commission), "commissionAsset": asset}
        order = {
            "symbol": "BTCUSDT", "orderId": order_id, "clientOrderId": client_id,
            "transactTime": NOW_MS, "updateTime": NOW_MS, "status": "FILLED", "side": side,
            "executedQty": str(quantity), "cummulativeQuoteQty": str(quote), "fills": [fill],
        }
        self.orders[client_id] = order
        self.trades[order_id] = [fill]
        if self.timeout_after_placing:
            self.timeout_after_placing = False
            raise BinanceOrderUnknownError("simulated timeout after the order was placed")
        return 200, json.dumps(order).encode()


def make_client(exchange: FakeTestnet) -> BinanceTestnetClient:
    return BinanceTestnetClient(CREDENTIALS, transport=exchange, clock_ms=lambda: NOW_MS)


def make_engine(broker: TestnetBroker, cash: str = "1000") -> TradingEngine:
    return TradingEngine(BuyAndHold(), broker, Portfolio(Decimal(cash)), RiskManager(None, "1h"))


def test_signature_matches_binance_documentation_example() -> None:
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    query = (
        "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
        "&recvWindow=5000&timestamp=1499827319559"
    )
    assert sign(secret, query) == (
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
    )


@pytest.mark.parametrize("url", ["https://api.binance.com", "https://evil.example"])
def test_client_refuses_anything_but_the_testnet(url: str) -> None:
    with pytest.raises(BinanceTradingError, match="only the Spot Testnet"):
        BinanceTestnetClient(CREDENTIALS, base_url=url, transport=FakeTestnet())


def test_credentials_come_only_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY", raising=False)
    with pytest.raises(BinanceTradingError, match="BINANCE_TESTNET_API_KEY"):
        Credentials.from_environment()
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "k")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "s")
    assert Credentials.from_environment() == Credentials("k", "s")


def test_time_sync_offsets_signed_timestamps() -> None:
    exchange = FakeTestnet()
    client = make_client(exchange)
    client.sync_time()
    client.balances()
    assert exchange.requests[-1] == ("GET", "/api/v3/account")


def test_buy_commission_in_btc_keeps_the_portfolio_equal_to_exchange_balances() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t1")
    bar = bars_from_closes([100])[0]
    portfolio = Portfolio(Decimal(10000))

    (fill,) = broker.submit(Order(Side.BUY, Decimal("2.000004"), bar.open_time), bar)
    portfolio.apply(fill)

    # 2.00000 BTC after lot-size rounding, minus 0.1% commission taken in BTC.
    assert fill.quantity == Decimal("1.998")
    assert fill.price == Decimal(100)
    assert fill.fee == Decimal("0.2")
    assert portfolio.quantity == exchange.balances["BTC"] - Decimal(1)
    assert portfolio.cash == exchange.balances["USDT"]


def test_orders_below_exchange_minimums_are_not_sent() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    bar = bars_from_closes([100])[0]

    assert broker.submit(Order(Side.BUY, Decimal("0.01"), bar.open_time), bar) == []
    assert ("POST", "/api/v3/order") not in exchange.requests


def test_a_timed_out_order_is_found_instead_of_sent_twice() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    bar = bars_from_closes([100])[0]
    exchange.timeout_after_placing = True

    (fill,) = broker.submit(Order(Side.BUY, Decimal(1), bar.open_time), bar)

    assert len(exchange.orders) == 1
    assert fill.client_order_id in exchange.orders
    assert broker.in_flight_order_id is None


def test_rejected_orders_return_no_fills() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    bar = bars_from_closes([100])[0]

    assert broker.submit(Order(Side.BUY, Decimal(1000), bar.open_time), bar) == []


def test_in_flight_order_is_recovered_after_a_crash() -> None:
    exchange = FakeTestnet()
    bar = bars_from_closes([100])[0]
    crashed = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    persisted: list[dict[str, str | None]] = []
    crashed.on_state_change = lambda: persisted.append(crashed.snapshot())
    # The order executes, but the response is lost and the follow-up lookup
    # fails too, so the outcome is unknown when the process dies.
    exchange.timeout_after_placing = True
    exchange.fail_lookups = True

    with pytest.raises(BinanceOrderUnknownError):
        crashed.submit(Order(Side.BUY, Decimal(1), bar.open_time), bar)
    assert persisted[-1]["in_flight_order_id"] in exchange.orders

    exchange.fail_lookups = False
    restarted = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    restarted.restore(persisted[-1])

    (fill,) = restarted.recover_in_flight_order()
    assert fill.side is Side.BUY
    assert fill.quantity == Decimal("0.999")
    assert restarted.in_flight_order_id is None
    assert len(exchange.orders) == 1


def test_reconciliation_sets_a_baseline_then_trusts_the_exchange() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    engine = make_engine(broker)

    assert reconcile_with_exchange(engine, broker) == []
    assert broker.baseline_base_balance == Decimal(1)

    exchange.balances["BTC"] += Decimal("0.5")  # e.g. a fill the bot never recorded
    warnings = reconcile_with_exchange(engine, broker)
    assert "position mismatch" in warnings[0]
    assert engine.portfolio.quantity == Decimal("0.5")


def test_reconciliation_halts_when_the_account_was_reset() -> None:
    exchange = FakeTestnet()
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="t")
    engine = make_engine(broker)
    reconcile_with_exchange(engine, broker)

    exchange.balances["BTC"] = Decimal("0.2")
    reconcile_with_exchange(engine, broker)

    assert engine.risk.halted


def test_live_trader_runs_on_the_testnet_broker(tmp_path: Path) -> None:
    bars = market(80)
    clock = Clock(after_close(bars[29]))
    exchange = FakeTestnet(price=bars[30].close)
    broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="run")
    store = TradingStore(tmp_path / "db.sqlite")
    trader = LiveTrader(
        store=store,
        feed=ClosedBarFeed(FakeExchange(bars, clock), "BTC/USDT", "1h", clock=clock),
        broker=broker,
        settings=TraderSettings(
            mode="testnet",
            symbol="BTC/USDT",
            timeframe="1h",
            strategy="buy_and_hold",
            initial_cash=Decimal(1000),
            warmup_bars=30,
            backfill_missed_bars=False,
            engine_config=EngineConfig(),
        ),
        run_id="testnet-run",
        on_start=lambda engine: reconcile_with_exchange(engine, broker),
    )
    trader.start()
    clock.now = after_close(bars[30])
    trader.step()

    (fill,) = store.fills("testnet-run")
    assert fill.side is Side.BUY
    assert exchange.balances["BTC"] - broker.baseline_base_balance == fill.quantity  # type: ignore[operator]
    assert trader.engine.portfolio.cash <= Decimal(1000) - fill.notional  # type: ignore[union-attr]

    # Restarting reconciles cleanly: the stored position matches the exchange.
    clock.now += timedelta(minutes=1)
    resumed_broker = TestnetBroker(make_client(exchange), "BTC/USDT", run_tag="run")
    warnings: list[str] = []
    resumed = LiveTrader(
        store=store,
        feed=ClosedBarFeed(FakeExchange(bars, clock), "BTC/USDT", "1h", clock=clock),
        broker=resumed_broker,
        settings=trader.settings,
        run_id="testnet-run",
        on_start=lambda engine: warnings.extend(reconcile_with_exchange(engine, resumed_broker)),
    )
    resumed.start()
    assert warnings == []
    assert resumed.engine.portfolio.quantity == fill.quantity  # type: ignore[union-attr]


def test_testnet_script_trades_the_fresh_bar_within_its_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import testnet_trade

    bars = market(40)
    clock = Clock(after_close(bars[30]))
    exchange = FakeTestnet(price=bars[30].close)
    monkeypatch.setattr(testnet_trade, "LOG_DIR", tmp_path / "logs")
    original_init = ClosedBarFeed.__init__

    def init_with_clock(self: ClosedBarFeed, *args: object, **kwargs: object) -> None:
        kwargs["clock"] = clock
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ClosedBarFeed, "__init__", init_with_clock)
    database = tmp_path / "db.sqlite"
    arguments = [
        "--strategy", "buy_and_hold", "--warmup-bars", "30", "--initial-cash", "1000",
        "--db", str(database), "--kill-file", str(tmp_path / "KILL"), "--run-id", "tn", "--once",
    ]

    assert testnet_trade.main(arguments, client=make_client(exchange),
                              provider=FakeExchange(bars, clock)) == 0
    clock.now = after_close(bars[31])
    assert testnet_trade.main(arguments, client=make_client(exchange),
                              provider=FakeExchange(bars, clock)) == 0

    (fill,) = TradingStore(database).fills("tn")
    assert fill.side is Side.BUY
    # The bot never spends more than its 1,000 USDT allocation, whatever the price did.
    assert Decimal(10000) - exchange.balances["USDT"] <= Decimal(1000)
    assert exchange.balances["USDT"] == Decimal(10000) - fill.notional - fill.fee
