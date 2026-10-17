"""A long-running trader that feeds live closed bars through the trading engine.

Paper trading uses the backtest's :class:`SimulatedBroker`; the testnet uses
an exchange broker.  Either way, the strategy, risk limits, and order logic
are the exact code that was backtested.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.core.engine import EngineConfig, TradingEngine
from src.core.health import Heartbeat, write_heartbeat
from src.core.portfolio import Portfolio
from src.data.models import OHLCVCandle
from src.database.store import RunRecord, TradingStore
from src.execution.costs import CostModel
from src.execution.interfaces import Broker
from src.execution.models import OrderReason
from src.risk.manager import RiskConfig, RiskManager
from src.strategies.registry import strategy_factory

from .feed import ClosedBarFeed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TraderSettings:
    """Everything that defines a run; a resumed run must use the same settings."""

    mode: str
    symbol: str
    timeframe: str
    strategy: str
    parameters: dict[str, Any] = field(default_factory=dict)
    initial_cash: Decimal = Decimal(10000)
    cost_model: CostModel = field(default_factory=CostModel)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    warmup_bars: int = 500
    backfill_missed_bars: bool = True

    def fingerprint(self) -> dict[str, Any]:
        """Settings that must match when resuming a run, as JSON-safe values."""

        return {
            "cost_model": {name: str(value) for name, value in _fields(self.cost_model).items()},
            "risk_config": {name: _json(value) for name, value in _fields(self.risk_config).items()},
            "engine_config": {name: str(value) for name, value in _fields(self.engine_config).items()},
            "warmup_bars": self.warmup_bars,
        }


def settings_from_record(record: RunRecord) -> TraderSettings:
    """Rebuild the settings a stored run was created with."""

    saved = record.settings
    risk_values: dict[str, Any] = {}
    for item in fields(RiskConfig):
        value = saved["risk_config"][item.name]
        if value is not None and "Decimal" in str(item.type):
            value = Decimal(value)
        risk_values[item.name] = value
    return TraderSettings(
        mode=record.mode,
        symbol=record.symbol,
        timeframe=record.timeframe,
        strategy=record.strategy,
        parameters=dict(record.parameters),
        initial_cash=record.initial_cash,
        cost_model=CostModel(
            **{name: Decimal(value) for name, value in saved["cost_model"].items()}
        ),
        risk_config=RiskConfig(**risk_values),
        engine_config=EngineConfig(
            **{name: Decimal(value) for name, value in saved["engine_config"].items()}
        ),
        warmup_bars=int(saved["warmup_bars"]),
    )


class RunConflictError(RuntimeError):
    """Raised when resuming a run with different settings than it was created with."""


class LiveTrader:
    """Process closed bars as they arrive, persisting everything after each bar.

    * On start, recent history is replayed without trading to rebuild strategy
      and indicator state; a resumed run then restores its saved portfolio,
      risk-limit, and pending-order state.
    * Bars missed while stopped are traded on resume (``backfill_missed_bars``)
      for paper trading, which keeps paper results identical to a backtest.
      Exchange brokers cannot trade the past; see :meth:`_catch_up`.
    * If ``kill_file`` exists, the position is closed immediately and the run
      halts permanently; the halt survives restarts.
    """

    def __init__(
        self,
        *,
        store: TradingStore,
        feed: ClosedBarFeed,
        broker: Broker,
        settings: TraderSettings,
        run_id: str,
        heartbeat_path: Path | None = None,
        kill_file: Path | None = None,
        on_start: Callable[[TradingEngine], None] | None = None,
        max_signal_age: timedelta = timedelta(minutes=10),
    ) -> None:
        self.store = store
        self.max_signal_age = max_signal_age
        self.feed = feed
        self.broker = broker
        self.settings = settings
        self.run_id = run_id
        self.heartbeat_path = heartbeat_path
        self.kill_file = kill_file
        self._on_start = on_start
        self.engine: TradingEngine | None = None
        self.last_bar: OHLCVCandle | None = None
        self._recorded_fills = 0
        self._recorded_equity = 0

    def start(self) -> None:
        settings = self.settings
        self._register_run()
        self.engine = TradingEngine(
            strategy=strategy_factory(settings.strategy, settings.parameters)(),
            broker=self.broker,
            portfolio=Portfolio(cash=settings.initial_cash),
            risk=RiskManager(settings.risk_config, settings.timeframe),
            config=settings.engine_config,
        )
        if hasattr(self.broker, "on_state_change"):
            # Exchange brokers persist an order's ID before sending it.
            self.broker.on_state_change = self._persist_state
        saved = self.store.load_state(self.run_id)
        # A resumed run needs a full warm-up *before* its last processed bar,
        # plus every bar that closed while it was stopped.
        missed_count = 0 if saved is None else self.feed.bars_closed_since(saved.last_bar_time)
        history = self.feed.history(settings.warmup_bars + missed_count)

        missed: list[OHLCVCandle] = []
        if saved is None:
            with self.store.transaction():
                for bar in history:
                    self.engine.process_bar(bar, trading=False)
                    self.store.record_bar(self.run_id, bar, traded=False)
                self.last_bar = history[-1] if history else None
                self._save_state()
            self._event("INFO", f"started new {settings.mode} run with {len(history)} warm-up bars")
        else:
            replay = [bar for bar in history if bar.open_time <= saved.last_bar_time]
            missed = [bar for bar in history if bar.open_time > saved.last_bar_time]
            for bar in replay:
                self.engine.process_bar(bar, trading=False)
            self.engine.restore(saved.snapshot["engine"])
            restore = getattr(self.broker, "restore", None)
            if restore is not None:
                restore(saved.snapshot.get("broker", {}))
            self.last_bar = replay[-1] if replay else None
            self._event(
                "INFO",
                f"resumed run after {saved.last_bar_time.isoformat()}; "
                f"{len(missed)} bar(s) closed while stopped",
            )

        # Reconcile with the exchange before trading anything.
        if self._on_start is not None:
            self._on_start(self.engine)
            with self.store.transaction():
                self._record_new_activity()
                self._save_state()
        if missed:
            self._catch_up(missed)
        self._write_heartbeat("halted" if self.engine.risk.halted else "running")

    def _catch_up(self, missed: list[OHLCVCandle]) -> None:
        """Handle bars that closed while the trader was stopped.

        Paper runs trade them all (the simulated broker fills at their historical
        opens).  Exchange runs cannot trade the past: stale bars only update
        indicators, and the latest bar is traded only if it closed within
        ``max_signal_age``, so its decision is still current.
        """

        engine = self._require_engine()
        if self.settings.backfill_missed_bars:
            self._process(missed)
            return
        *stale, latest = missed
        fresh = self.feed.now() - latest.close_time <= self.max_signal_age
        for bar in stale if fresh else missed:
            engine.process_bar(bar, trading=False)
            self.last_bar = bar
        with self.store.transaction():
            self._save_state()
        skipped = len(stale) if fresh else len(missed)
        if skipped:
            self._event("WARNING", f"did not trade {skipped} stale bar(s) missed while stopped")
        if fresh:
            self._process([latest])

    def step(self) -> int:
        """Process any newly closed bars; return how many were processed."""

        engine = self._require_engine()
        if self._check_kill_switch():
            return 0
        if self.last_bar is None:
            bars = self.feed.history(1)
        else:
            bars = self.feed.poll(self.last_bar.open_time)
        self._process(bars)
        self._write_heartbeat("halted" if engine.risk.halted else "running")
        return len(bars)

    def run_forever(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        max_iterations: int | None = None,
        retry_interval: float = 15.0,
    ) -> None:
        """Poll until stopped, sleeping until each next bar should have closed.

        Errors (for example an exchange timeout) are logged and retried after
        ``retry_interval`` seconds instead of stopping the trader.
        """

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                self.step()
            except Exception as error:
                LOGGER.exception("step failed")
                self._event("ERROR", f"step failed: {error}")
                self._write_heartbeat("error", str(error))
                sleep(retry_interval)
                continue
            wait = 0.0
            if self.last_bar is not None:
                wait = self.feed.seconds_until_next_close(self.last_bar.open_time)
            sleep(wait if wait > 0 else retry_interval)

    def _process(self, bars: list[OHLCVCandle]) -> None:
        engine = self._require_engine()
        for bar in bars:
            engine.process_bar(bar, trading=True)
            self.last_bar = bar
            with self.store.transaction():
                self.store.record_bar(self.run_id, bar, traded=True)
                self._record_new_activity()
                self._save_state()
            LOGGER.info(
                "bar %s close=%s equity=%s position=%s",
                bar.open_time.isoformat(),
                bar.close,
                engine.state.equity_curve[-1].equity,
                engine.portfolio.quantity,
            )

    def _record_new_activity(self) -> None:
        engine = self._require_engine()
        for fill in engine.state.fills[self._recorded_fills :]:
            self.store.record_fill(self.run_id, fill)
            LOGGER.info(
                "fill %s %s @ %s fee=%s reason=%s",
                fill.side.value,
                fill.quantity,
                fill.price,
                fill.fee,
                fill.reason.value,
            )
        self._recorded_fills = len(engine.state.fills)
        for point in engine.state.equity_curve[self._recorded_equity :]:
            self.store.record_equity(self.run_id, point)
        self._recorded_equity = len(engine.state.equity_curve)

    def _save_state(self) -> None:
        engine = self._require_engine()
        if self.last_bar is None:
            return
        snapshot_broker = getattr(self.broker, "snapshot", None)
        self.store.save_state(
            self.run_id,
            self.last_bar.open_time,
            {
                "engine": engine.snapshot(),
                "broker": snapshot_broker() if snapshot_broker is not None else {},
            },
        )

    def _persist_state(self) -> None:
        with self.store.transaction():
            self._save_state()

    def _check_kill_switch(self) -> bool:
        engine = self._require_engine()
        if self.kill_file is None or not self.kill_file.exists():
            return False
        if not engine.risk.halted or engine.portfolio.quantity > 0:
            engine.risk.halt(f"manual kill switch {self.kill_file} at {datetime.now(UTC).isoformat()}")
            cancel = getattr(self.broker, "cancel_pending", None)
            if cancel is not None:
                cancel()
            if engine.portfolio.quantity > 0 and self.last_bar is not None:
                fills = self.broker.close_at(
                    self.last_bar, engine.portfolio.quantity, OrderReason.RISK_LIMIT
                )
                with self.store.transaction():
                    engine.apply_closing_fills(fills, self.last_bar)
                    self._record_new_activity()
                    self._save_state()
            else:
                with self.store.transaction():
                    self._save_state()
            self._event("CRITICAL", "kill switch engaged; position closed and trading halted")
        self._write_heartbeat("halted", "kill switch engaged")
        return True

    def _register_run(self) -> None:
        settings = self.settings
        existing = self.store.get_run(self.run_id)
        fingerprint = settings.fingerprint()
        if existing is None:
            self.store.create_run(
                RunRecord(
                    run_id=self.run_id,
                    mode=settings.mode,
                    symbol=settings.symbol,
                    timeframe=settings.timeframe,
                    strategy=settings.strategy,
                    parameters=dict(settings.parameters),
                    settings=fingerprint,
                    initial_cash=settings.initial_cash,
                    created_at=datetime.now(UTC),
                )
            )
            return
        current = (settings.mode, settings.symbol, settings.timeframe, settings.strategy,
                   dict(settings.parameters), fingerprint, settings.initial_cash)
        saved = (existing.mode, existing.symbol, existing.timeframe, existing.strategy,
                 existing.parameters, existing.settings, existing.initial_cash)
        if current != saved:
            raise RunConflictError(
                f"run {self.run_id!r} was created with different settings; "
                "use a new --run-id or the original settings"
            )

    def _event(self, level: str, message: str) -> None:
        LOGGER.log(getattr(logging, level), message)
        self.store.log_event(self.run_id, level, message)

    def _write_heartbeat(self, status: str, message: str = "") -> None:
        if self.heartbeat_path is None or self.engine is None:
            return
        curve = self.engine.state.equity_curve
        write_heartbeat(
            self.heartbeat_path,
            Heartbeat(
                run_id=self.run_id,
                status=status,
                updated_at=datetime.now(UTC).isoformat(),
                last_bar_time=None if self.last_bar is None else self.last_bar.open_time.isoformat(),
                equity=str(curve[-1].equity) if curve else None,
                quantity=str(self.engine.portfolio.quantity),
                halted=self.engine.risk.halted,
                message=message,
            ),
        )

    def _require_engine(self) -> TradingEngine:
        if self.engine is None:
            raise RuntimeError("call start() before processing bars")
        return self.engine


def _fields(instance: object) -> dict[str, object]:
    return {name: getattr(instance, name) for name in instance.__slots__}  # type: ignore[attr-defined]


def _json(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
