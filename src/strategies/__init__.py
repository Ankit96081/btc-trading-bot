"""Trading strategies, indicators, and the strategy registry."""

from .base import FLAT, FULLY_LONG, Strategy, StrategyFactory
from .baselines import BuyAndHold, DonchianBreakout, MovingAverageCrossover
from .registry import (
    STRATEGIES,
    StrategyConfigurationError,
    strategy_factory,
    strategy_parameters,
)
from .trend import MultiTimeframeTrend, TrendFilter

__all__ = [
    "FLAT",
    "FULLY_LONG",
    "STRATEGIES",
    "BuyAndHold",
    "DonchianBreakout",
    "MovingAverageCrossover",
    "MultiTimeframeTrend",
    "Strategy",
    "StrategyConfigurationError",
    "StrategyFactory",
    "TrendFilter",
    "strategy_factory",
    "strategy_parameters",
]
