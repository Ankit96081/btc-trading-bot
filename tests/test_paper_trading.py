"""Tests for the live bar feed, persistence, and the paper trader (no network)."""

import math
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from src.backtest import run_backtest
from src.core.health import is_stale, read_heartbeat
from src.data.models import CandleRequest, OHLCVCandle
from src.database import RunRecord, TradingStore
from src.execution import CostModel, OrderReason, SimulatedBroker
from src.paper import ClosedBarFeed, LiveTrader, RunConflictError, TraderSettings
from src.risk import RiskConfig
from src.strategies import strategy_factory
from tests.factories import bars_from_closes

WARMUP = 30
# slow_period close to WARMUP makes restarts fail loudly if the replay is too short.
PARAMETERS = {"fast_period": 5, "slow_period": 28}
RISK = RiskConfig(stop_loss_fraction=Decimal("0.03"), max_drawdown=Decimal("0.5"))


def market(count: int = 200, seed: int = 3) -> list[OHLCVCandle]:
    rng = random.Random(seed)
    price, closes = 100.0, []
    for index in range(count):
        drift = 0.003 if (index // 40) % 2 == 0 else -0.003
        price *= math.exp(drift + rng.gauss(0, 0.006))
        closes.append(round(price, 2))
    return bars_from_closes(closes, spread="0.3")


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeExchange:
    """Serves bars that have opened by the clock's time, like Binance does."""

    def __init__(self, bars: list[OHLCVCandle], clock: Clock) -> None:
        self._bars = bars
        self._clock = clock
        self.fail_next = False

    def get_candles(self, request: CandleRequest) -> list[OHLCVCandle]:
        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("simulated exchange timeout")
        assert request.start_time is not None and request.end_time is not None
        visible = [
            bar
            for bar in self._bars
            if request.start_time <= bar.open_time <= request.end_time
            and bar.open_time <= self._clock()
        ]
        return visible[: request.limit]


def after_close(bar: OHLCVCandle) -> datetime:
    return bar.close_time + timedelta(seconds=10)


def settings(**changes: object) -> TraderSettings:
    values: dict[str, object] = {
        "mode": "paper",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "strategy": "ma_crossover",
        "parameters": PARAMETERS,
        "risk_config": RISK,
        "warmup_bars": WARMUP,
    }
    values.update(changes)
    return TraderSettings(**values)  # type: ignore[arg-type]


def make_trader(
    store: TradingStore, bars: list[OHLCVCandle], clock: Clock, **options: object
) -> LiveTrader:
    exchange = FakeExchange(bars, clock)
    feed = ClosedBarFeed(exchange, "BTC/USDT", "1h", clock=clock)
    trader = LiveTrader(
        store=store,
        feed=feed,
        broker=SimulatedBroker(CostModel()),
        settings=options.pop("settings", settings()),  # type: ignore[arg-type]
        run_id="test-run",
        **options,  # type: ignore[arg-type]
    )
    trader.exchange = exchange  # type: ignore[attr-defined]
    return trader


def expected_backtest(bars: list[OHLCVCandle], until: int):
    return run_backtest(
        bars[:until],
        strategy_factory("ma_crossover", PARAMETERS),
        cost_model=CostModel(),
        risk_config=RISK,
        warmup_bars=WARMUP,
    )


def drive(trader: LiveTrader, clock: Clock, bars: list[OHLCVCandle], start: int, stop: int) -> None:
    for index in range(start, stop):
        clock.now = after_close(bars[index])
        trader.step()


def test_feed_never_returns_the_bar_that_is_still_forming() -> None:
    bars = market(10)
    clock = Clock(bars[5].open_time + timedelta(minutes=30))
    feed = ClosedBarFeed(FakeExchange(bars, clock), "BTC/USDT", "1h", clock=clock)

    assert [bar.open_time for bar in feed.history(3)] == [b.open_time for b in bars[2:5]]
    assert feed.poll(bars[3].open_time) == [bars[4]]
    assert feed.poll(bars[4].open_time) == []


def test_paper_trading_matches_the_backtest_exactly(tmp_path: Path) -> None:
    bars = market()
    clock = Clock(after_close(bars[WARMUP - 1]))
    store = TradingStore(tmp_path / "paper.db")
    trader = make_trader(store, bars, clock)
    trader.start()

    drive(trader, clock, bars, WARMUP, len(bars))

    expected = expected_backtest(bars, len(bars))
    assert expected.fills, "the scenario should trade"
    assert store.fills("test-run") == list(expected.fills)
    assert store.equity_curve("test-run") == list(expected.equity_curve)


def test_restart_and_downtime_do_not_change_results(tmp_path: Path) -> None:
    bars = market()
    clock = Clock(after_close(bars[WARMUP - 1]))
    database = tmp_path / "paper.db"
    first = make_trader(TradingStore(database), bars, clock)
    first.start()
    drive(first, clock, bars, WARMUP, 90)
    first.store.close()

    # Stopped while bars 90-95 closed, then restarted from the database.
    clock.now = after_close(bars[95])
    store = TradingStore(database)
    second = make_trader(store, bars, clock)
    second.start()
    drive(second, clock, bars, 96, len(bars))

    expected = expected_backtest(bars, len(bars))
    assert store.fills("test-run") == list(expected.fills)
    assert store.equity_curve("test-run") == list(expected.equity_curve)
    assert any("6 bar(s) closed while stopped" in event[2] for event in store.events("test-run"))


def test_resuming_with_different_settings_is_refused(tmp_path: Path) -> None:
    bars = market(60)
    clock = Clock(after_close(bars[WARMUP - 1]))
    database = tmp_path / "paper.db"
    make_trader(TradingStore(database), bars, clock).start()

    changed = settings(parameters={"fast_period": 6, "slow_period": 28})
    with pytest.raises(RunConflictError):
        make_trader(TradingStore(database), bars, clock, settings=changed).start()


def test_kill_switch_flattens_and_halts_across_restarts(tmp_path: Path) -> None:
    bars = market()
    clock = Clock(after_close(bars[WARMUP - 1]))
    database = tmp_path / "paper.db"
    kill_file = tmp_path / "KILL"
    trader = make_trader(TradingStore(database), bars, clock, kill_file=kill_file)
    trader.start()
    # Advance until a position is open.
    index = WARMUP
    while trader.engine.portfolio.quantity == 0:  # type: ignore[union-attr]
        drive(trader, clock, bars, index, index + 1)
        index += 1

    kill_file.touch()
    trader.step()

    assert trader.engine.portfolio.quantity == 0  # type: ignore[union-attr]
    assert trader.store.fills("test-run")[-1].reason is OrderReason.RISK_LIMIT
    trader.store.close()

    kill_file.unlink()
    store = TradingStore(database)
    resumed = make_trader(store, bars, clock)
    resumed.start()
    fills_before = len(store.fills("test-run"))
    drive(resumed, clock, bars, index, len(bars))
    assert resumed.engine.risk.halted  # type: ignore[union-attr]
    assert len(store.fills("test-run")) == fills_before


def test_run_forever_survives_exchange_errors_and_writes_heartbeats(tmp_path: Path) -> None:
    bars = market(80)
    clock = Clock(after_close(bars[WARMUP - 1]))
    heartbeat = tmp_path / "heartbeat.json"
    trader = make_trader(TradingStore(tmp_path / "paper.db"), bars, clock, heartbeat_path=heartbeat)
    trader.start()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += timedelta(hours=1)

    trader.exchange.fail_next = True  # type: ignore[attr-defined]
    trader.run_forever(sleep=fake_sleep, max_iterations=5, retry_interval=15)

    assert sleeps[0] == 15
    events = trader.store.events("test-run")
    assert any(level == "ERROR" and "timeout" in message for _, level, message in events)
    beat = read_heartbeat(heartbeat)
    assert beat.status == "running"
    assert beat.last_bar_time is not None
    assert not is_stale(heartbeat, timedelta(minutes=5))
    assert is_stale(tmp_path / "missing.json", timedelta(minutes=5))


def test_store_round_trips_bars_with_their_traded_flag(tmp_path: Path) -> None:
    bars = market(3)
    store = TradingStore(tmp_path / "db.sqlite")
    store.create_run(
        RunRecord(
            run_id="r",
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1h",
            strategy="buy_and_hold",
            parameters={},
            settings={},
            initial_cash=Decimal(1),
            created_at=bars[0].open_time,
        )
    )
    with store.transaction():
        store.record_bar("r", bars[0], traded=False)
        store.record_bar("r", bars[1], traded=True)

    assert store.bars("r") == [(bars[0], False), (bars[1], True)]
    assert store.get_run("r").initial_cash == Decimal(1)  # type: ignore[union-attr]


def test_paper_trade_and_compare_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import compare_paper_backtest, paper_trade

    bars = market()
    clock = Clock(after_close(bars[WARMUP - 1]))
    monkeypatch.setattr(paper_trade, "BinanceSpotMarketDataProvider", lambda: FakeExchange(bars, clock))
    monkeypatch.setattr(paper_trade, "LOG_DIR", tmp_path / "logs")
    database = tmp_path / "trading.db"
    arguments = [
        "--strategy", "ma_crossover",
        "--param", "fast_period=5",
        "--param", "slow_period=28",
        "--warmup-bars", str(WARMUP),
        "--stop-pct", "0.03",
        "--max-drawdown", "0.5",
        "--db", str(database),
        "--kill-file", str(tmp_path / "KILL"),
        "--run-id", "script-run",
        "--once",
    ]

    # ClosedBarFeed captured the real clock as a default argument, so patch its instances.
    original_init = ClosedBarFeed.__init__

    def init_with_clock(self: ClosedBarFeed, *args: object, **kwargs: object) -> None:
        kwargs["clock"] = clock
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ClosedBarFeed, "__init__", init_with_clock)
    assert paper_trade.main(arguments) == 0
    for index in range(WARMUP, len(bars)):
        clock.now = after_close(bars[index])
        assert paper_trade.main(arguments) == 0

    assert compare_paper_backtest.main(["--db", str(database), "--run-id", "script-run"]) == 0
    output = capsys.readouterr().out
    assert "Every fill matches the backtest exactly." in output
    assert (tmp_path / "logs" / "script-run.heartbeat.json").exists()
