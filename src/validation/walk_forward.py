"""Walk-forward optimization: tune on the past, measure on the unseen future."""

import itertools
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.runner import BacktestResult, run_backtest
from src.core.engine import EngineConfig, EquityPoint, RoundTrip
from src.data.models import OHLCVCandle
from src.execution.costs import CostModel
from src.execution.models import Fill
from src.risk.manager import RiskConfig
from src.strategies.registry import StrategyConfigurationError, strategy_factory

ParameterSet = dict[str, object]

RISK_PARAMETER_PREFIX = "risk."
RISK_PARAMETERS: dict[str, tuple[str, Callable[[object], object]]] = {
    "risk.vol_target": ("volatility_target", float),
    "risk.vol_lookback": ("volatility_lookback", int),
    "risk.stop_atr": ("stop_loss_atr_multiple", float),
}
"""Risk settings a grid may tune alongside strategy parameters, e.g. ``risk.vol_target``."""


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Window sizes in bars.

    Rolling windows keep ``train_bars`` fixed; anchored windows always start
    at the first bar.  ``warmup_bars`` of history before each test window prime
    indicators without trading, so the test window's first bar already has
    formed signals.  ``min_train_round_trips`` rejects parameter sets that
    barely traded in-sample, whose scores are mostly noise.
    """

    train_bars: int
    test_bars: int
    anchored: bool = False
    warmup_bars: int = 500
    min_train_round_trips: int = 5

    def __post_init__(self) -> None:
        if self.train_bars <= 0 or self.test_bars <= 0 or self.warmup_bars < 0:
            raise ValueError("train_bars and test_bars must be positive; warmup_bars >= 0")


@dataclass(frozen=True, slots=True)
class FoldResult:
    index: int
    train_start: OHLCVCandle
    test_start: OHLCVCandle
    test_end: OHLCVCandle
    parameters: ParameterSet
    train_score: float
    neighbor_ratio: float | None
    test: BacktestResult

    @property
    def test_metrics(self) -> PerformanceMetrics:
        return self.test.metrics


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    strategy: str
    folds: tuple[FoldResult, ...]
    out_of_sample: PerformanceMetrics
    initial_cash: Decimal

    @property
    def profitable_fold_fraction(self) -> float:
        profitable = sum(1 for fold in self.folds if fold.test_metrics.total_return > 0)
        return profitable / len(self.folds)

    @property
    def median_neighbor_ratio(self) -> float | None:
        ratios = [fold.neighbor_ratio for fold in self.folds if fold.neighbor_ratio is not None]
        return statistics.median(ratios) if ratios else None

    @property
    def final_parameters(self) -> ParameterSet:
        return self.folds[-1].parameters


def split_parameters(
    parameters: Mapping[str, object], risk_config: RiskConfig | None
) -> tuple[ParameterSet, RiskConfig | None]:
    """Separate ``risk.*`` entries from strategy parameters and apply them to ``risk_config``."""

    strategy_parameters: ParameterSet = {}
    overrides: dict[str, object] = {}
    for name, value in parameters.items():
        if not name.startswith(RISK_PARAMETER_PREFIX):
            strategy_parameters[name] = value
            continue
        if name not in RISK_PARAMETERS:
            known = ", ".join(RISK_PARAMETERS)
            raise StrategyConfigurationError(
                f"unknown risk parameter {name!r}; choose from {known}"
            )
        field, convert = RISK_PARAMETERS[name]
        try:
            overrides[field] = convert(value)
        except (TypeError, ValueError) as error:
            raise StrategyConfigurationError(f"invalid {name} value {value!r}") from error
    if not overrides:
        return strategy_parameters, risk_config
    return strategy_parameters, replace(risk_config or RiskConfig(), **overrides)


def expand_grid(grid: Mapping[str, Sequence[object]]) -> list[ParameterSet]:
    """Return every combination of the grid's values, in a deterministic order."""

    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


def run_walk_forward(
    bars: Sequence[OHLCVCandle],
    strategy: str,
    grid: Mapping[str, Sequence[object]],
    config: WalkForwardConfig,
    *,
    initial_cash: Decimal = Decimal(10000),
    cost_model: CostModel | None = None,
    risk_config: RiskConfig | None = None,
    engine_config: EngineConfig | None = None,
    fixed_parameters: Sequence[ParameterSet] | None = None,
) -> WalkForwardResult:
    """Run walk-forward optimization of ``strategy`` over ``grid``.

    Each fold selects the parameter set with the best train-window Sharpe
    ratio and trades it on the following test window, starting with the
    previous fold's ending equity.  Positions are closed, with costs, at the
    end of every test window so no fold inherits another's position.  Grid
    entries named in :data:`RISK_PARAMETERS` tune ``risk_config`` per fold
    instead of the strategy.  Pass ``fixed_parameters`` (one set per
    fold, for example from an earlier run) to skip selection; the cost stress
    test uses this to re-run the exact same choices under harsher costs.
    """

    windows = _fold_windows(len(bars), config)
    if not windows:
        raise ValueError("not enough bars for one train window plus one test window")
    if fixed_parameters is not None and len(fixed_parameters) != len(windows):
        raise ValueError("fixed_parameters must provide one parameter set per fold")

    candidates = expand_grid(grid)
    folds: list[FoldResult] = []
    cash = initial_cash
    for index, (train_start, test_start, test_end) in enumerate(windows):
        if fixed_parameters is not None:
            parameters, score, neighbor_ratio = dict(fixed_parameters[index]), math.nan, None
        else:
            parameters, score, neighbor_ratio = _select_parameters(
                bars[train_start:test_start],
                strategy,
                candidates,
                grid,
                config,
                cost_model,
                risk_config,
                engine_config,
            )
        warmup_start = max(0, test_start - config.warmup_bars)
        strategy_parameters, fold_risk_config = split_parameters(parameters, risk_config)
        test = run_backtest(
            bars[warmup_start:test_end],
            strategy_factory(strategy, strategy_parameters),
            initial_cash=cash,
            cost_model=cost_model,
            risk_config=fold_risk_config,
            engine_config=engine_config,
            warmup_bars=test_start - warmup_start,
            liquidate_at_end=True,
        )
        cash = test.equity_curve[-1].equity
        folds.append(
            FoldResult(
                index=index,
                train_start=bars[train_start],
                test_start=bars[test_start],
                test_end=bars[test_end - 1],
                parameters=parameters,
                train_score=score,
                neighbor_ratio=neighbor_ratio,
                test=test,
            )
        )

    return WalkForwardResult(
        strategy=strategy,
        folds=tuple(folds),
        out_of_sample=_stitch(folds, initial_cash),
        initial_cash=initial_cash,
    )


def _fold_windows(bar_count: int, config: WalkForwardConfig) -> list[tuple[int, int, int]]:
    windows: list[tuple[int, int, int]] = []
    test_start = config.train_bars
    while test_start + config.test_bars <= bar_count:
        train_start = 0 if config.anchored else test_start - config.train_bars
        windows.append((train_start, test_start, test_start + config.test_bars))
        test_start += config.test_bars
    return windows


def _select_parameters(
    train: Sequence[OHLCVCandle],
    strategy: str,
    candidates: list[ParameterSet],
    grid: Mapping[str, Sequence[object]],
    config: WalkForwardConfig,
    cost_model: CostModel | None,
    risk_config: RiskConfig | None,
    engine_config: EngineConfig | None,
) -> tuple[ParameterSet, float, float | None]:
    scores: dict[int, float] = {}
    for position, parameters in enumerate(candidates):
        try:
            strategy_parameters, candidate_risk_config = split_parameters(parameters, risk_config)
            factory = strategy_factory(strategy, strategy_parameters)
        except (StrategyConfigurationError, ValueError):
            continue
        result = run_backtest(
            train,
            factory,
            cost_model=cost_model,
            risk_config=candidate_risk_config,
            engine_config=engine_config,
        )
        if len(result.round_trips) < config.min_train_round_trips:
            continue
        scores[position] = result.metrics.sharpe_ratio

    if not scores:
        raise ValueError(
            "no parameter set traded enough in a train window; widen the grid, "
            "lengthen train windows, or lower min_train_round_trips"
        )
    best = max(scores, key=lambda position: (scores[position], -position))
    return candidates[best], scores[best], _neighbor_ratio(best, scores, candidates, grid)


def _neighbor_ratio(
    best: int,
    scores: Mapping[int, float],
    candidates: list[ParameterSet],
    grid: Mapping[str, Sequence[object]],
) -> float | None:
    """Mean score of grid neighbours (every index within one step) relative to the best."""

    best_score = scores[best]
    if best_score <= 0:
        return None

    def indices(parameters: ParameterSet) -> tuple[int, ...]:
        return tuple(list(values).index(parameters[name]) for name, values in grid.items())

    best_indices = indices(candidates[best])
    neighbor_scores = []
    for position, parameters in enumerate(candidates):
        if position == best:
            continue
        distance = max(abs(a - b) for a, b in zip(indices(parameters), best_indices, strict=True))
        if distance == 1:
            # Invalid or untraded neighbours count as a score of zero: a spike
            # surrounded by unusable settings is exactly what this detects.
            neighbor_scores.append(scores.get(position, 0.0))
    if not neighbor_scores:
        return None
    return statistics.mean(neighbor_scores) / best_score


def _stitch(folds: Sequence[FoldResult], initial_cash: Decimal) -> PerformanceMetrics:
    curve: list[EquityPoint] = []
    fills: list[Fill] = []
    trips: list[RoundTrip] = []
    for fold in folds:
        curve.extend(fold.test.equity_curve)
        fills.extend(fold.test.fills)
        trips.extend(fold.test.round_trips)
    first = folds[0].test
    return compute_metrics(
        curve,
        fills,
        trips,
        timeframe=first.timeframe,
        initial_equity=initial_cash,
        benchmark_start_price=first.first_open,
    )


def fold_parameters(result: WalkForwardResult) -> list[ParameterSet]:
    """Return the parameters each fold selected, for re-running identical choices."""

    return [fold.parameters for fold in result.folds]
