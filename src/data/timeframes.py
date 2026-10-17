"""Timeframe strings such as ``1m`` or ``4h`` and their durations."""

import re
from datetime import timedelta

_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([mhd])$")
_UNIT_TO_MINUTES = {"m": 1, "h": 60, "d": 1_440}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Return the bar duration for a timeframe such as ``15m``, ``1h`` or ``1d``."""

    return timedelta(minutes=timeframe_to_minutes(timeframe))


def timeframe_to_minutes(timeframe: str) -> int:
    """Return the number of whole minutes in a timeframe."""

    match = _TIMEFRAME_PATTERN.match(timeframe) if isinstance(timeframe, str) else None
    if match is None:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use forms like 1m, 15m, 1h or 1d")
    return int(match.group(1)) * _UNIT_TO_MINUTES[match.group(2)]


def bars_per_year(timeframe: str) -> float:
    """Return the number of bars in a 365-day year; crypto markets trade 24/7."""

    return 365 * 1_440 / timeframe_to_minutes(timeframe)
