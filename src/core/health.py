"""Heartbeat files that let an external monitor see whether a trader is alive."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Heartbeat:
    run_id: str
    status: str
    updated_at: str
    last_bar_time: str | None
    equity: str | None
    quantity: str | None
    halted: bool
    message: str = ""


def write_heartbeat(path: Path, heartbeat: Heartbeat) -> None:
    """Atomically replace the heartbeat file so readers never see a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(asdict(heartbeat), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_heartbeat(path: Path) -> Heartbeat:
    return Heartbeat(**json.loads(path.read_text(encoding="utf-8")))


def heartbeat_age(path: Path, now: datetime | None = None) -> timedelta:
    beat = read_heartbeat(path)
    return (now or datetime.now(UTC)) - datetime.fromisoformat(beat.updated_at)


def is_stale(path: Path, max_age: timedelta, now: datetime | None = None) -> bool:
    """Return True when the heartbeat is missing, unreadable, or older than ``max_age``."""

    try:
        return heartbeat_age(path, now) > max_age
    except (OSError, ValueError, TypeError, KeyError):
        return True
