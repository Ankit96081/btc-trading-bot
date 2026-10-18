"""Tests for the higher-timeframe trend strategies."""

from decimal import Decimal

import pytest

from src.strategies import (
    FLAT,
    FULLY_LONG,
    MultiTimeframeTrend,
    Strategy,
    TrendFilter,
    strategy_factory,
)
from src.strategies.trend import PeriodCloses
from tests.factories import bars_from_closes


def hourly_days(*daily_closes: float) -> list[float]:
    """Twenty-four identical hourly closes per day."""

    return [close for close in daily_closes for _ in range(24)]


def test_period_closes_report_only_the_last_bar_of_each_period() -> None:
    closes = PeriodCloses(240)
    bars = bars_from_closes(list(range(1, 9)))

    reported = [closes.update(bar) for bar in bars]

    assert reported == [None, None, None, 4.0, None, None, None, 8.0]


def test_trend_filter_signals_only_at_day_closes() -> None:
    strategy = TrendFilter(sma_days=2)
    assert isinstance(strategy, Strategy)

    signals = [strategy.on_bar(bar) for bar in bars_from_closes(hourly_days(10, 10, 12, 9))]

    assert all(signal is None for index, signal in enumerate(signals) if index % 24 != 23)
    # Day 2: close equals the SMA -> flat; day 3: above -> long; day 4: below -> flat.
    assert [signals[index] for index in (23, 47, 71, 95)] == [None, FLAT, FULLY_LONG, FLAT]


def test_trend_filter_band_requires_a_clear_break() -> None:
    strategy = TrendFilter(sma_days=2, band=0.1)

    signals = [strategy.on_bar(bar) for bar in bars_from_closes(hourly_days(10, 10, 12, 15))]

    # 12 is not above 11 * 1.1 = 12.1; 15 is above 13.5 * 1.1 = 14.85.
    assert [signals[index] for index in (47, 71, 95)] == [FLAT, FLAT, FULLY_LONG]


def test_multi_timeframe_trend_scales_exposure_with_agreeing_timeframes() -> None:
    rising = [100 + index for index in range(48)]
    bars = bars_from_closes([*rising, 50])

    strategy = MultiTimeframeTrend(ema_period=2)
    signals = [strategy.on_bar(bar) for bar in bars]

    # Nothing until the daily EMA forms at the second day's close.
    assert all(signal is None for signal in signals[:47])
    assert signals[47] == FULLY_LONG
    # A sharp hourly drop flips only the 1h vote: 4h and 1d have not closed.
    assert signals[48] == Decimal(2) / Decimal(3)


def test_multi_timeframe_trend_min_votes_requires_agreement() -> None:
    bars = bars_from_closes([*(100 + index for index in range(48)), 50])
    strategy = MultiTimeframeTrend(ema_period=2, min_votes=3)

    signals = [strategy.on_bar(bar) for bar in bars]

    assert signals[47] == FULLY_LONG
    assert signals[48] == FLAT


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("trend_filter", {"band": "1.5"}),
        ("multi_timeframe_trend", {"min_votes": "4"}),
        ("multi_timeframe_trend", {"min_votes": "0"}),
    ],
)
def test_trend_strategies_reject_invalid_parameters(
    name: str, parameters: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        strategy_factory(name, parameters)


def test_trend_strategies_build_from_string_parameters() -> None:
    assert isinstance(strategy_factory("trend_filter", {"sma_days": "50", "band": "0.02"})(), TrendFilter)
    assert isinstance(
        strategy_factory("multi_timeframe_trend", {"ema_period": "20", "min_votes": "2"})(),
        MultiTimeframeTrend,
    )
