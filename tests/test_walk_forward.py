"""Tests for walk-forward validation and the predefined criteria."""

import math
import random
from dataclasses import replace
from decimal import Decimal

import pytest

from src.backtest import run_backtest
from src.data.models import OHLCVCandle
from src.execution import CostModel, OrderReason
from src.strategies import BuyAndHold
from src.validation import (
    HoldoutCriteria,
    ValidationCriteria,
    ValidationEvidence,
    WalkForwardConfig,
    expand_grid,
    fold_parameters,
    run_walk_forward,
)
from src.validation.walk_forward import _fold_windows
from tests.factories import bars_from_closes

GRID = {"fast_period": [5, 10], "slow_period": [30, 60]}
CONFIG = WalkForwardConfig(train_bars=600, test_bars=200, warmup_bars=100, min_train_round_trips=2)


def trending_market(seed: int = 7, bars: int = 2_000) -> list[OHLCVCandle]:
    """Long up and down trends with mild noise: trend following has a real edge here."""

    rng = random.Random(seed)
    price, closes = 100.0, []
    for index in range(bars):
        drift = 0.004 if (index // 150) % 2 == 0 else -0.003
        price *= math.exp(drift + rng.gauss(0, 0.002))
        closes.append(round(price, 4))
    return bars_from_closes(closes, spread="0.05")


def random_walk(seed: int = 11, bars: int = 2_000) -> list[OHLCVCandle]:
    rng = random.Random(seed)
    price, closes = 100.0, []
    for _ in range(bars):
        price *= math.exp(rng.gauss(0, 0.01))
        closes.append(round(price, 4))
    return bars_from_closes(closes, spread="0.05")


def evidence_for(bars: list[OHLCVCandle]) -> ValidationEvidence:
    result = run_walk_forward(bars, "ma_crossover", GRID, CONFIG)
    stressed = run_walk_forward(
        bars,
        "ma_crossover",
        GRID,
        CONFIG,
        cost_model=CostModel().scaled(2),
        fixed_parameters=fold_parameters(result),
    )
    return ValidationEvidence(
        out_of_sample=result.out_of_sample,
        profitable_fold_fraction=result.profitable_fold_fraction,
        stressed_sharpe_ratio=stressed.out_of_sample.sharpe_ratio,
        median_neighbor_ratio=result.median_neighbor_ratio,
    )


def test_rolling_and_anchored_fold_windows() -> None:
    config = WalkForwardConfig(train_bars=10, test_bars=5)
    assert _fold_windows(22, config) == [(0, 10, 15), (5, 15, 20)]
    anchored = replace(config, anchored=True)
    assert _fold_windows(22, anchored) == [(0, 10, 15), (0, 15, 20)]
    assert _fold_windows(14, config) == []


def test_expand_grid_is_deterministic() -> None:
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_test_windows_are_contiguous_and_chained() -> None:
    bars = trending_market()

    result = run_walk_forward(bars, "ma_crossover", GRID, CONFIG)

    assert len(result.folds) == 7
    for previous, current in zip(result.folds, result.folds[1:], strict=False):
        assert current.test_start.open_time > previous.test_end.open_time
        assert current.test.initial_cash == previous.test.equity_curve[-1].equity
        assert previous.test.final_quantity == 0
    stitched_bars = sum(len(fold.test.equity_curve) for fold in result.folds)
    assert result.out_of_sample.bars == stitched_bars == 7 * 200


def test_walk_forward_detects_a_real_edge_out_of_sample() -> None:
    evidence = evidence_for(trending_market())

    metrics = evidence.out_of_sample
    assert metrics.total_return > 0
    assert metrics.sharpe_ratio > 1
    assert evidence.profitable_fold_fraction >= 0.6


def test_walk_forward_does_not_validate_noise() -> None:
    results = ValidationCriteria(min_round_trips=5).evaluate(evidence_for(random_walk()))

    assert not all(result.passed for result in results)
    assert not next(r for r in results if r.name == "OOS Sharpe ratio").passed


def test_fixed_parameters_reproduce_the_same_choices() -> None:
    bars = trending_market()
    first = run_walk_forward(bars, "ma_crossover", GRID, CONFIG)

    again = run_walk_forward(
        bars, "ma_crossover", GRID, CONFIG, fixed_parameters=fold_parameters(first)
    )

    assert again.out_of_sample == first.out_of_sample


def test_walk_forward_requires_enough_data() -> None:
    with pytest.raises(ValueError, match="not enough bars"):
        run_walk_forward(trending_market(bars=700), "ma_crossover", GRID, CONFIG)


def test_liquidate_at_end_sells_at_the_last_close_with_costs() -> None:
    bars = bars_from_closes([100, 100, 110])
    costs = CostModel(fee_rate=Decimal("0.001"), slippage_bps=Decimal(0))

    result = run_backtest(bars, BuyAndHold, cost_model=costs, liquidate_at_end=True)

    final = result.fills[-1]
    assert final.reason is OrderReason.END_OF_PERIOD
    assert final.price == Decimal(110)
    assert result.final_quantity == 0
    buy = result.fills[0]
    expected_cash = Decimal(10000) - buy.notional - buy.fee + final.notional - final.fee
    assert result.equity_curve[-1].equity == expected_cash


def test_criteria_report_each_check() -> None:
    evidence = evidence_for(trending_market())

    results = ValidationCriteria().evaluate(evidence)

    assert [result.name for result in results] == [
        "OOS total return",
        "OOS Sharpe ratio",
        "OOS Sharpe vs buy-and-hold",
        "OOS max drawdown",
        "Profitable folds",
        "OOS round trips",
        "Fills per day",
        "Sharpe with 2x costs",
        "Parameter robustness",
    ]
    holdout = HoldoutCriteria().evaluate(evidence.out_of_sample)
    assert [result.name for result in holdout] == ["Holdout total return", "Holdout max drawdown"]
