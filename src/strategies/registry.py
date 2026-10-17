"""Look up strategies by name and build them from loosely typed parameters."""

import inspect
from collections.abc import Mapping
from functools import partial

from .base import Strategy, StrategyFactory
from .baselines import BuyAndHold, DonchianBreakout, MovingAverageCrossover

STRATEGIES: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHold,
    "ma_crossover": MovingAverageCrossover,
    "donchian_breakout": DonchianBreakout,
}


class StrategyConfigurationError(ValueError):
    """Raised when a strategy name or its parameters are invalid."""


def strategy_factory(name: str, parameters: Mapping[str, object] | None = None) -> StrategyFactory:
    """Return a factory for ``name`` configured with ``parameters``.

    String values (for example from a command line) are converted to the type
    of the parameter's default value.  The factory is validated by building
    one instance immediately, so bad parameters fail before any backtest runs.
    """

    try:
        strategy_class = STRATEGIES[name]
    except KeyError as error:
        known = ", ".join(sorted(STRATEGIES))
        raise StrategyConfigurationError(f"unknown strategy {name!r}; choose from {known}") from error

    defaults = strategy_parameters(name)
    converted: dict[str, object] = {}
    for key, value in (parameters or {}).items():
        if key not in defaults:
            known = ", ".join(defaults) or "none"
            raise StrategyConfigurationError(
                f"{name} has no parameter {key!r}; parameters: {known}"
            )
        converted[key] = _convert(key, value, defaults[key])

    factory = partial(strategy_class, **converted)
    try:
        factory()
    except (TypeError, ValueError) as error:
        raise StrategyConfigurationError(f"invalid {name} parameters: {error}") from error
    return factory


def strategy_parameters(name: str) -> dict[str, object]:
    """Return a strategy's tunable parameters and their default values."""

    signature = inspect.signature(STRATEGIES[name].__init__)
    return {
        parameter.name: parameter.default
        for parameter in signature.parameters.values()
        if parameter.name != "self" and parameter.default is not inspect.Parameter.empty
    }


def _convert(key: str, value: object, default: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        if isinstance(default, bool):
            lowered = value.strip().lower()
            if lowered not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(value)
            return lowered in {"true", "1", "yes"}
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
    except ValueError as error:
        raise StrategyConfigurationError(
            f"parameter {key!r} expects {type(default).__name__}, got {value!r}"
        ) from error
    return value
