"""Tests for the trading engine, simulated execution, risk limits, and metrics."""

import itertools
import math
import statistics
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal

import pytest

from src.backtest import compute_metrics, run_backtest
from src.core import EngineConfig, EquityPoint, InsufficientFundsError, Portfolio
from src.data.models import OHLCVCandle
from src.execution import CostModel, Fill, OrderReason, Side
from src.risk import RiskConfig, RiskManager
from src.strategies import BuyAndHold
from tests.factories import DEFAULT_START, bars_from_closes, make_bar

NO_COSTS = CostModel(fee_rate=Decimal(0), slippage_bps=Decimal(0))
FEES_ONLY = CostModel(fee_rate=Decimal("0.001"), slippage_bps=Decimal(0))
STEP = Decimal("0.00001")


class ScriptedStrategy:
    """Returns a fixed sequence of targets, one per bar, and records what it saw."""

    def __init__(self, targets: Sequence[str | None]) -> None:
        self._targets = list(targets)
        self.seen: list[OHLCVCandle] = []

    def on_bar(self, bar: OHLCVCandle) -> Decimal | None:
        self.seen.append(bar)
        index = len(self.seen) - 1
        value = self._targets[index] if index < len(self._targets) else None
        return None if value is None else Decimal(value)


def scripted(targets: Sequence[str | None]):
    return lambda: ScriptedStrategy(targets)


def hourly(open_time_offset: int, **prices: float) -> OHLCVCandle:
    return make_bar(
        DEFAULT_START + timedelta(hours=open_time_offset), timeframe="1h", **prices
    )


def test_orders_fill_at_the_next_bar_open_never_the_decision_bar() -> None:
    bars = bars_from_closes([100, 110, 121])

    result = run_backtest(bars, BuyAndHold, cost_model=NO_COSTS)

    assert result.equity_curve[0].equity == Decimal(10000)
    assert result.equity_curve[0].quantity == 0
    (fill,) = result.fills
    assert fill.timestamp == bars[1].open_time
    assert fill.price == bars[1].open


def test_a_single_round_trip_pays_both_fees_exactly() -> None:
    bars = bars_from_closes([100, 110, 120, 125])

    result = run_backtest(bars, scripted(["1", None, "0"]), cost_model=FEES_ONLY)

    buy, sell = result.fills
    quantity = (Decimal(10000) / (Decimal(100) * Decimal("1.001")) / STEP).to_integral_value(
        rounding="ROUND_DOWN"
    ) * STEP
    assert (buy.side, buy.price, buy.quantity) == (Side.BUY, Decimal(100), quantity)
    assert (sell.side, sell.price, sell.quantity) == (Side.SELL, Decimal(120), quantity)
    expected_pnl = quantity * Decimal(120) * Decimal("0.999") - quantity * Decimal(100) * Decimal(
        "1.001"
    )
    (trip,) = result.round_trips
    assert trip.pnl == expected_pnl
    assert result.equity_curve[-1].equity == Decimal(10000) + expected_pnl
    assert result.final_quantity == 0


def test_slippage_moves_fills_against_the_trader() -> None:
    bars = bars_from_closes([100, 100, 100, 100])
    costs = CostModel(fee_rate=Decimal(0), slippage_bps=Decimal(10))

    result = run_backtest(bars, scripted(["1", None, "0"]), cost_model=costs)

    buy, sell = result.fills
    assert buy.price == Decimal("100.1")
    assert sell.price == Decimal("99.9")


def test_warmup_bars_prime_the_strategy_without_trading() -> None:
    bars = bars_from_closes([100, 101, 102, 103])
    strategy = ScriptedStrategy(["1", "1", "1", "1"])

    result = run_backtest(bars, lambda: strategy, cost_model=NO_COSTS, warmup_bars=2)

    assert len(strategy.seen) == 4
    assert len(result.equity_curve) == 2
    assert result.fills[0].timestamp == bars[3].open_time
    assert result.first_open == bars[2].open


def test_stop_loss_exits_at_the_stop_price_and_waits_for_a_flat_signal() -> None:
    bars = [
        hourly(0, open=100, high=100, low=100, close=100),
        hourly(1, open=100, high=101, low=99, close=100),  # buy fills at 100
        hourly(2, open=100, high=100, low=94, close=96),  # stop at 95 is hit
        hourly(3, open=96, high=97, low=95, close=96),  # still long signal: no re-entry
        hourly(4, open=96, high=97, low=95, close=96),  # flat signal re-arms
        hourly(5, open=96, high=97, low=95, close=96),  # long again -> re-enter
        hourly(6, open=96, high=97, low=95, close=96),
    ]
    risk = RiskConfig(stop_loss_fraction=Decimal("0.05"))

    result = run_backtest(
        bars, scripted(["1", "1", "1", "1", "0", "1"]), cost_model=NO_COSTS, risk_config=risk
    )

    _entry, stop, reentry = result.fills
    assert stop.reason is OrderReason.STOP_LOSS
    assert stop.price == Decimal(95)
    assert stop.timestamp == bars[2].open_time
    assert result.round_trips[0].exit_reason is OrderReason.STOP_LOSS
    assert reentry.timestamp == bars[6].open_time


def test_stop_loss_fills_at_the_open_when_price_gaps_through_it() -> None:
    bars = [
        hourly(0, open=100, high=100, low=100, close=100),
        hourly(1, open=100, high=100, low=100, close=100),
        hourly(2, open=90, high=91, low=89, close=90),
    ]
    risk = RiskConfig(stop_loss_fraction=Decimal("0.05"))

    result = run_backtest(bars, scripted(["1"]), cost_model=NO_COSTS, risk_config=risk)

    assert result.fills[-1].price == Decimal(90)


def test_daily_loss_limit_flattens_and_locks_until_the_next_utc_day() -> None:
    day_one = [
        hourly(0, open=100, high=100, low=100, close=100),
        hourly(1, open=100, high=100, low=100, close=100),  # buy fills
        hourly(2, open=100, high=100, low=95, close=95),  # -5% -> forced exit decided
        hourly(3, open=95, high=96, low=94, close=95),  # exit fills; stays locked
    ]
    day_two = [hourly(24, open=95, high=96, low=94, close=95)]  # unlocked -> re-entry decided
    day_two.append(hourly(25, open=95, high=96, low=94, close=95))
    risk = RiskConfig(max_daily_loss=Decimal("0.03"))

    result = run_backtest(
        day_one + day_two, BuyAndHold, cost_model=NO_COSTS, risk_config=risk
    )

    _entry, exit_fill, reentry = result.fills
    assert exit_fill.reason is OrderReason.RISK_LIMIT
    assert exit_fill.timestamp == day_one[3].open_time
    assert reentry.timestamp == day_two[1].open_time


def test_max_drawdown_kill_switch_halts_permanently() -> None:
    bars = bars_from_closes([100, 100, 120, 80, 90, 100, 150])
    risk = RiskConfig(max_drawdown=Decimal("0.2"))

    result = run_backtest(bars, BuyAndHold, cost_model=NO_COSTS, risk_config=risk)

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]
    assert result.fills[-1].reason is OrderReason.RISK_LIMIT
    assert result.final_quantity == 0
    assert result.halt_reason is not None and "max drawdown" in result.halt_reason


def test_trade_cap_blocks_new_entries_but_not_exits() -> None:
    bars = bars_from_closes([100] * 8)
    risk = RiskConfig(max_trades_per_day=2)

    result = run_backtest(
        bars, scripted(["1", "0", "1", "0", "1"]), cost_model=NO_COSTS, risk_config=risk
    )

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]


def test_small_rebalances_are_suppressed_but_large_ones_execute() -> None:
    bars = bars_from_closes([100] * 6)

    result = run_backtest(
        bars,
        scripted(["1", "0.98", "0.5"]),
        cost_model=NO_COSTS,
        engine_config=EngineConfig(min_rebalance_exposure=Decimal("0.05")),
    )

    _buy, sell = result.fills
    assert sell.side is Side.SELL
    assert sell.quantity == Decimal(50)
    assert result.round_trips == ()


def test_buy_quantity_is_reduced_when_the_price_gaps_up_before_the_fill() -> None:
    bars = [
        hourly(0, open=100, high=100, low=100, close=100),
        hourly(1, open=110, high=110, low=110, close=110),
    ]

    result = run_backtest(bars, BuyAndHold, cost_model=FEES_ONLY)

    (fill,) = result.fills
    assert fill.notional + fill.fee <= Decimal(10000)
    assert fill.quantity < Decimal(100)


def test_volatility_targeting_scales_exposure() -> None:
    manager = RiskManager(RiskConfig(volatility_target=0.2, volatility_lookback=4), "1h")
    closes = [100, 101, 100, 101, 100]
    for bar in bars_from_closes(closes):
        manager.observe_bar(bar)

    log_returns = [math.log(b / a) for a, b in itertools.pairwise(closes)]
    annualized = statistics.stdev(log_returns) * math.sqrt(8760)
    expected = Decimal(str(0.2 / annualized))

    assert manager.target_exposure(Decimal(1)) == min(expected, Decimal(1))
    assert manager.target_exposure(Decimal(0)) == 0


def test_volatility_targeting_stays_flat_until_warm() -> None:
    manager = RiskManager(RiskConfig(volatility_target=0.2, volatility_lookback=10), "1h")
    manager.observe_bar(bars_from_closes([100])[0])
    assert manager.target_exposure(Decimal(1)) == 0


@pytest.mark.parametrize(
    "config",
    [
        {"max_exposure": Decimal(2)},
        {"max_daily_loss": Decimal(0)},
        {"max_drawdown": Decimal(1)},
        {"max_trades_per_day": 0},
        {"volatility_target": -1.0},
    ],
)
def test_risk_config_rejects_invalid_limits(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RiskConfig(**config)  # type: ignore[arg-type]


def test_portfolio_rejects_overspending_and_overselling() -> None:
    portfolio = Portfolio(cash=Decimal(100))
    with pytest.raises(InsufficientFundsError):
        portfolio.apply(Fill(DEFAULT_START, Side.BUY, Decimal(2), Decimal(60), Decimal(0)))
    with pytest.raises(InsufficientFundsError):
        portfolio.apply(Fill(DEFAULT_START, Side.SELL, Decimal(1), Decimal(60), Decimal(0)))


def test_metrics_match_hand_calculations() -> None:
    equities = [110, 99, 121]
    curve = [
        EquityPoint(DEFAULT_START + timedelta(hours=index), Decimal(price), Decimal(value), Decimal(1))
        for index, (value, price) in enumerate(zip(equities, [11, 9.9, 12.1], strict=False))
    ]

    metrics = compute_metrics(
        curve,
        [],
        [],
        timeframe="1h",
        initial_equity=Decimal(100),
        benchmark_start_price=Decimal(10),
    )

    returns = [0.1, -0.1, 121 / 99 - 1]
    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.max_drawdown == pytest.approx(0.1)
    assert metrics.sharpe_ratio == pytest.approx(
        statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(8760)
    )
    assert metrics.benchmark_return == pytest.approx(0.21)
    assert metrics.exposure_time == 1.0
    assert metrics.annualized_return == pytest.approx(1.21 ** (8760 / 3) - 1)
