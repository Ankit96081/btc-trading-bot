"""Exit non-zero when a trader looks unhealthy, for cron or monitoring alerts.

Example (alert if the hourly trader has been silent for more than 2 hours):
    python scripts/check_health.py logs/paper-ma_crossover-1h.heartbeat.json --max-age-minutes 120

Exit codes: 0 healthy, 1 stale or unreadable heartbeat, 2 trader reports an
error, 3 trading halted (kill switch or risk limit).
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.health import heartbeat_age, is_stale, read_heartbeat


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heartbeat", type=Path)
    parser.add_argument("--max-age-minutes", type=float, default=120)
    args = parser.parse_args(arguments)

    max_age = timedelta(minutes=args.max_age_minutes)
    if is_stale(args.heartbeat, max_age):
        print(f"UNHEALTHY: heartbeat {args.heartbeat} is missing or older than {max_age}")
        return 1
    beat = read_heartbeat(args.heartbeat)
    age = heartbeat_age(args.heartbeat)
    summary = (f"run={beat.run_id} status={beat.status} last_bar={beat.last_bar_time} "
               f"equity={beat.equity} position={beat.quantity} age={age}")
    if beat.status == "error":
        print(f"UNHEALTHY: {summary} message={beat.message}")
        return 2
    if beat.halted:
        print(f"HALTED: {summary} message={beat.message}")
        return 3
    print(f"OK: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
