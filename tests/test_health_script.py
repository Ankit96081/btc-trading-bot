"""Tests for the heartbeat health-check script."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import check_health
from src.core.health import Heartbeat, write_heartbeat


def beat(path: Path, *, status: str = "running", halted: bool = False, age_minutes: int = 0) -> None:
    updated = datetime.now(UTC) - timedelta(minutes=age_minutes)
    write_heartbeat(
        path,
        Heartbeat("run", status, updated.isoformat(), None, "100", "0", halted, "note"),
    )


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({}, 0),
        ({"age_minutes": 200}, 1),
        ({"status": "error"}, 2),
        ({"status": "halted", "halted": True}, 3),
    ],
)
def test_check_health_exit_codes(tmp_path: Path, options: dict[str, object], code: int) -> None:
    path = tmp_path / "heartbeat.json"
    beat(path, **options)  # type: ignore[arg-type]

    assert check_health.main([str(path), "--max-age-minutes", "120"]) == code


def test_missing_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    assert check_health.main([str(tmp_path / "missing.json")]) == 1
