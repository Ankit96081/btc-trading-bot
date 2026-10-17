"""SQLite persistence for paper and testnet trading runs.

Every processed bar, fill, and equity mark is stored so that a run can be
resumed after a restart and later compared with a backtest over the same
bars.  Decimal values are stored as text to keep them exact.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.core.engine import EquityPoint
from src.data.csv_format import candle_to_csv_row, csv_row_to_candle
from src.data.models import OHLCVCandle
from src.execution.models import Fill, OrderReason, Side

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,
    parameters TEXT NOT NULL,
    settings TEXT NOT NULL,
    initial_cash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bars (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    open_time TEXT NOT NULL,
    candle TEXT NOT NULL,
    traded INTEGER NOT NULL,
    PRIMARY KEY (run_id, open_time)
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    fee TEXT NOT NULL,
    reason TEXT NOT NULL,
    client_order_id TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp TEXT NOT NULL,
    close TEXT NOT NULL,
    equity TEXT NOT NULL,
    quantity TEXT NOT NULL,
    PRIMARY KEY (run_id, timestamp)
);
CREATE TABLE IF NOT EXISTS state (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    last_bar_time TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    mode: str
    symbol: str
    timeframe: str
    strategy: str
    parameters: dict[str, Any]
    settings: dict[str, Any]
    initial_cash: Decimal
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SavedState:
    last_bar_time: datetime
    snapshot: dict[str, Any]


class TradingStore:
    """A small repository over one SQLite file; safe to reopen after a crash."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.execute("PRAGMA foreign_keys = ON")
        # WAL with NORMAL sync never corrupts the database; at worst a power
        # cut loses the last bar's commit, which is re-processed on restart.
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group writes so that a bar's fills, equity, and state commit together."""

        with self._connection:
            yield

    def create_run(self, run: RunRecord) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.mode,
                    run.symbol,
                    run.timeframe,
                    run.strategy,
                    json.dumps(run.parameters, sort_keys=True),
                    json.dumps(run.settings, sort_keys=True),
                    str(run.initial_cash),
                    run.created_at.isoformat(),
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row[0],
            mode=row[1],
            symbol=row[2],
            timeframe=row[3],
            strategy=row[4],
            parameters=json.loads(row[5]),
            settings=json.loads(row[6]),
            initial_cash=Decimal(row[7]),
            created_at=datetime.fromisoformat(row[8]),
        )

    def record_bar(self, run_id: str, bar: OHLCVCandle, *, traded: bool) -> None:
        """Store a processed bar; ``traded=False`` marks warm-up bars."""

        self._connection.execute(
            "INSERT OR REPLACE INTO bars VALUES (?, ?, ?, ?)",
            (run_id, bar.open_time.isoformat(), json.dumps(candle_to_csv_row(bar)), int(traded)),
        )

    def record_fill(self, run_id: str, fill: Fill) -> None:
        self._connection.execute(
            "INSERT INTO fills (run_id, timestamp, side, quantity, price, fee, reason, "
            "client_order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                fill.timestamp.isoformat(),
                fill.side.value,
                str(fill.quantity),
                str(fill.price),
                str(fill.fee),
                fill.reason.value,
                fill.client_order_id,
            ),
        )

    def record_equity(self, run_id: str, point: EquityPoint) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                point.timestamp.isoformat(),
                str(point.close),
                str(point.equity),
                str(point.quantity),
            ),
        )

    def save_state(self, run_id: str, last_bar_time: datetime, snapshot: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO state VALUES (?, ?, ?, ?)",
            (
                run_id,
                last_bar_time.isoformat(),
                json.dumps(snapshot, sort_keys=True),
                datetime.now(UTC).isoformat(),
            ),
        )

    def load_state(self, run_id: str) -> SavedState | None:
        row = self._connection.execute(
            "SELECT last_bar_time, snapshot FROM state WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return SavedState(datetime.fromisoformat(row[0]), json.loads(row[1]))

    def log_event(self, run_id: str, level: str, message: str) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO events (run_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
                (run_id, datetime.now(UTC).isoformat(), level, message),
            )

    def bars(self, run_id: str) -> list[tuple[OHLCVCandle, bool]]:
        """Return ``(bar, traded)`` pairs in time order."""

        rows = self._connection.execute(
            "SELECT candle, traded FROM bars WHERE run_id = ? ORDER BY open_time", (run_id,)
        ).fetchall()
        return [(csv_row_to_candle(json.loads(row[0])), bool(row[1])) for row in rows]

    def fills(self, run_id: str) -> list[Fill]:
        rows = self._connection.execute(
            "SELECT timestamp, side, quantity, price, fee, reason, client_order_id "
            "FROM fills WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [
            Fill(
                timestamp=datetime.fromisoformat(row[0]),
                side=Side(row[1]),
                quantity=Decimal(row[2]),
                price=Decimal(row[3]),
                fee=Decimal(row[4]),
                reason=OrderReason(row[5]),
                client_order_id=row[6],
            )
            for row in rows
        ]

    def equity_curve(self, run_id: str) -> list[EquityPoint]:
        rows = self._connection.execute(
            "SELECT timestamp, close, equity, quantity FROM equity WHERE run_id = ? "
            "ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [
            EquityPoint(
                datetime.fromisoformat(row[0]), Decimal(row[1]), Decimal(row[2]), Decimal(row[3])
            )
            for row in rows
        ]

    def events(self, run_id: str) -> list[tuple[str, str, str]]:
        return [
            (row[0], row[1], row[2])
            for row in self._connection.execute(
                "SELECT timestamp, level, message FROM events WHERE run_id = ? ORDER BY id",
                (run_id,),
            )
        ]
