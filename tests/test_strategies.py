"""Tests for indicators, baseline strategies, and the strategy registry."""

from decimal import Decimal

import pytest

from src.strategies import (
    FLAT,
    FULLY_LONG,
    BuyAndHold,
    DonchianBreakout,
    MovingAverageCrossover,
    Strategy,
    StrategyConfigurationError,
    strategy_factory,
    strategy_parameters,
)
from src.strategies.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    RollingMaximum,
    RollingMinimum,
    RollingStandardDeviation,
    SimpleMovingAverage,
)
from tests.factories import DEFAULT_START, bars_from_closes, make_bar


def feed(indicator: object, values: list[float]) -> list[float | None]:
    return [indicator.update(value) for value in values]  # type: ignore[attr-defined]


def test_simple_moving_average() -> None:
    assert feed(SimpleMovingAverage(3), [1, 2, 3, 4]) == [None, None, 2, 3]


def test_exponential_moving_average_is_seeded_with_the_sma() -> None:
    assert feed(ExponentialMovingAverage(3), [1, 2, 3, 4]) == [None, None, 2, 3]
    assert feed(ExponentialMovingAverage(3), [1, 2, 3, 7])[-1] == 4.5


def test_rolling_standard_deviation() -> None:
    assert feed(RollingStandardDeviation(3), [1, 2, 3, 5]) == [
        None,
        None,
        1.0,
        pytest.approx(1.5275252),
    ]


def test_rolling_extremes() -> None:
    values = [3, 1, 2, 5, 1, 1, 1]
    assert feed(RollingMaximum(3), values) == [None, None, 3, 5, 5, 5, 1]
    assert feed(RollingMinimum(3), values) == [None, None, 1, 1, 1, 1, 1]


def test_average_true_range_uses_wilder_smoothing() -> None:
    atr = AverageTrueRange(2)
    assert atr.update(10, 8, 9) is None
    assert atr.update(12, 9, 11) == 2.5
    assert atr.update(11, 10, 10) == 1.75


def test_relative_strength_index() -> None:
    rsi = RelativeStrengthIndex(2)
    assert feed(rsi, [1, 2, 3, 2]) == [None, None, 100.0, 50.0]


@pytest.mark.parametrize("period", [0, -1, True])
def test_indicators_reject_invalid_periods(period: int) -> None:
    with pytest.raises(ValueError):
        SimpleMovingAverage(period)


def test_buy_and_hold_is_always_long() -> None:
    strategy = BuyAndHold()
    assert isinstance(strategy, Strategy)
    assert all(strategy.on_bar(bar) == FULLY_LONG for bar in bars_from_closes([1, 2, 1]))


def test_ma_crossover_waits_for_warmup_then_follows_the_trend() -> None:
    strategy = MovingAverageCrossover(fast_period=2, slow_period=3)
    closes = [10, 11, 12, 13, 12, 10, 8]

    signals = [strategy.on_bar(bar) for bar in bars_from_closes(closes)]

    assert signals == [None, None, FULLY_LONG, FULLY_LONG, FULLY_LONG, FLAT, FLAT]


def test_ma_crossover_rejects_inverted_periods() -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast_period=5, slow_period=5)


def test_donchian_breakout_compares_against_prior_bars_only() -> None:
    strategy = DonchianBreakout(entry_period=2, exit_period=2)
    start = DEFAULT_START
    bars = [
        make_bar(start, open=10, high=11, low=9, close=10, timeframe="1h"),
        make_bar(start.replace(hour=1), open=10, high=11, low=9, close=10, timeframe="1h"),
        # Closes at 11: equal to, not above, the prior 2-bar high -> no entry.
        make_bar(start.replace(hour=2), open=10, high=12, low=10, close=11, timeframe="1h"),
        # Closes above the prior high of 12 -> enter.
        make_bar(start.replace(hour=3), open=11, high=13, low=11, close=12.5, timeframe="1h"),
        # Stays above the prior 2-bar low of 10 -> hold.
        make_bar(start.replace(hour=4), open=12, high=12, low=10.5, close=11, timeframe="1h"),
        # Closes below the prior 2-bar low of 10.5 -> exit.
        make_bar(start.replace(hour=5), open=11, high=11, low=9, close=10, timeframe="1h"),
    ]

    signals = [strategy.on_bar(bar) for bar in bars]

    assert signals == [None, None, FLAT, FULLY_LONG, FULLY_LONG, FLAT]


def test_registry_builds_fresh_strategies_from_string_parameters() -> None:
    factory = strategy_factory("ma_crossover", {"fast_period": "3", "exponential": "true"})

    first, second = factory(), factory()

    assert isinstance(first, MovingAverageCrossover)
    assert first is not second


def test_registry_lists_tunable_parameters() -> None:
    assert strategy_parameters("donchian_breakout") == {"entry_period": 20, "exit_period": 10}


@pytest.mark.parametrize(
    ("name", "parameters", "message"),
    [
        ("nope", {}, "unknown strategy"),
        ("ma_crossover", {"window": "3"}, "no parameter"),
        ("ma_crossover", {"fast_period": "x"}, "expects int"),
        ("ma_crossover", {"fast_period": "60"}, "invalid ma_crossover parameters"),
    ],
)
def test_registry_rejects_bad_configuration(
    name: str, parameters: dict[str, object], message: str
) -> None:
    with pytest.raises(StrategyConfigurationError, match=message):
        strategy_factory(name, parameters)


def test_strategies_accept_decimal_prices() -> None:
    bar = make_bar(DEFAULT_START, open="1.1", high="1.2", low="1.0", close=Decimal("1.15"))
    assert DonchianBreakout().on_bar(bar) is None
