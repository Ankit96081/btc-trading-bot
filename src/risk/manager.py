"""Position sizing and protective limits applied on top of strategy signals.

The same risk manager runs in backtests, paper trading, and on the testnet,
so a limit that looks good in research behaves identically later on.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from src.data.models import OHLCVCandle
from src.data.timeframes import bars_per_year
from src.execution.models import Fill, OrderReason
from src.strategies.indicators import AverageTrueRange, RollingStandardDeviation


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk limits; every limit is disabled when set to ``None``.

    Fractions are of equity: ``max_daily_loss=Decimal("0.03")`` means 3%.
    """

    max_exposure: Decimal = Decimal(1)
    volatility_target: float | None = None
    volatility_lookback: int = 168
    stop_loss_atr_multiple: float | None = None
    stop_loss_fraction: Decimal | None = None
    atr_period: int = 14
    max_daily_loss: Decimal | None = None
    max_drawdown: Decimal | None = None
    max_trades_per_day: int | None = None

    def __post_init__(self) -> None:
        if not Decimal(0) <= self.max_exposure <= Decimal(1):
            raise ValueError("max_exposure must be between 0 and 1 for a spot account")
        for name in ("volatility_target", "stop_loss_atr_multiple"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("stop_loss_fraction", "max_daily_loss", "max_drawdown"):
            value = getattr(self, name)
            if value is not None and not Decimal(0) < value < Decimal(1):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_trades_per_day is not None and self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive")
        if self.volatility_lookback < 2 or self.atr_period < 1:
            raise ValueError("volatility_lookback must be >= 2 and atr_period >= 1")


class RiskManager:
    """Turns a strategy's desired exposure into a permitted exposure.

    Limits, in the order they are applied:

    * **Kill switch** - once equity falls ``max_drawdown`` below its peak the
      account is flattened and never trades again in this run.
    * **Daily loss limit** - once equity falls ``max_daily_loss`` below the
      start of the UTC day, the account is flattened until the next day.
    * **Stop re-arm** - after a stop-loss exit the strategy must signal flat
      before it may re-enter, so a stop is not immediately undone.
    * **Volatility targeting** - exposure is scaled so the position's
      annualized volatility approaches ``volatility_target``.
    * **Maximum exposure** cap.
    * **Trade cap** - once ``max_trades_per_day`` fills happened in a UTC day,
      only exits are allowed until the next day.
    """

    def __init__(self, config: RiskConfig | None, timeframe: str) -> None:
        self.config = config or RiskConfig()
        self._annualization = math.sqrt(bars_per_year(timeframe))
        self._atr = AverageTrueRange(self.config.atr_period)
        self._atr_value: float | None = None
        self._return_std = RollingStandardDeviation(self.config.volatility_lookback)
        self._volatility: float | None = None
        self._previous_close: float | None = None

        self._peak_equity: Decimal | None = None
        self._last_equity: Decimal | None = None
        self._day: date | None = None
        self._day_start_equity: Decimal | None = None
        self._trades_today = 0
        self._trade_day: date | None = None

        self.halted = False
        self.daily_locked = False
        self.halt_reason: str | None = None
        self._awaiting_flat_signal = False
        self.stop_price: Decimal | None = None

    def halt(self, reason: str) -> None:
        """Stop all new risk permanently, e.g. from a manual kill switch."""

        self.halted = True
        self.halt_reason = reason

    def snapshot(self) -> dict[str, str | int | bool | None]:
        """Return the limit state that must survive a restart.

        Indicator values are not included: they are rebuilt by replaying
        recent bars through :meth:`observe_bar`.
        """

        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "daily_locked": self.daily_locked,
            "awaiting_flat_signal": self._awaiting_flat_signal,
            "stop_price": _text(self.stop_price),
            "peak_equity": _text(self._peak_equity),
            "last_equity": _text(self._last_equity),
            "day": _text(self._day),
            "day_start_equity": _text(self._day_start_equity),
            "trades_today": self._trades_today,
            "trade_day": _text(self._trade_day),
        }

    def restore(self, snapshot: dict[str, str | int | bool | None]) -> None:
        self.halted = bool(snapshot["halted"])
        self.halt_reason = _optional_str(snapshot["halt_reason"])
        self.daily_locked = bool(snapshot["daily_locked"])
        self._awaiting_flat_signal = bool(snapshot["awaiting_flat_signal"])
        self.stop_price = _optional_decimal(snapshot["stop_price"])
        self._peak_equity = _optional_decimal(snapshot["peak_equity"])
        self._last_equity = _optional_decimal(snapshot["last_equity"])
        self._day = _optional_date(snapshot["day"])
        self._day_start_equity = _optional_decimal(snapshot["day_start_equity"])
        self._trades_today = int(snapshot["trades_today"] or 0)
        self._trade_day = _optional_date(snapshot["trade_day"])

    def observe_bar(self, bar: OHLCVCandle) -> None:
        """Update volatility estimates from a closed bar (also during warm-up)."""

        high, low, close = float(bar.high), float(bar.low), float(bar.close)
        self._atr_value = self._atr.update(high, low, close)
        if self._previous_close is not None and self._previous_close > 0 and close > 0:
            self._volatility = self._return_std.update(math.log(close / self._previous_close))
        self._previous_close = close

    def update_equity(self, timestamp: datetime, equity: Decimal) -> bool:
        """Record marked-to-market equity; return ``True`` if a limit now forces an exit."""

        today = timestamp.date()
        if today != self._day:
            self._day = today
            self._day_start_equity = self._last_equity if self._last_equity is not None else equity
            self.daily_locked = False
        self._last_equity = equity
        self._peak_equity = equity if self._peak_equity is None else max(self._peak_equity, equity)

        forced = False
        limit = self.config.max_drawdown
        if not self.halted and limit is not None and equity <= self._peak_equity * (1 - limit):
            self.halted = True
            self.halt_reason = f"max drawdown of {limit:%} breached at {timestamp.isoformat()}"
            forced = True
        limit = self.config.max_daily_loss
        if (
            not self.daily_locked
            and limit is not None
            and self._day_start_equity is not None
            and equity <= self._day_start_equity * (1 - limit)
        ):
            self.daily_locked = True
            forced = True
        return forced

    def target_exposure(self, strategy_target: Decimal) -> Decimal:
        """Return the exposure the account may hold given the strategy's wish."""

        if self.halted or self.daily_locked:
            return Decimal(0)
        if self._awaiting_flat_signal:
            if strategy_target > 0:
                return Decimal(0)
            self._awaiting_flat_signal = False

        target = strategy_target
        if self.config.volatility_target is not None and target > 0:
            if not self._volatility:
                return Decimal(0)
            annualized = self._volatility * self._annualization
            scale = Decimal(str(self.config.volatility_target / annualized))
            target *= scale
        return max(Decimal(0), min(target, self.config.max_exposure))

    def can_add_risk(self, timestamp: datetime) -> bool:
        """Return whether a risk-increasing fill is allowed under the daily trade cap."""

        cap = self.config.max_trades_per_day
        if cap is None:
            return True
        return self._trade_count(timestamp.date()) < cap

    def record_fill(self, fill: Fill, position_before: Decimal, position_after: Decimal) -> None:
        """Track fills for the trade cap, stop placement, and stop re-arming."""

        self._trades_today = self._trade_count(fill.timestamp.date()) + 1

        if position_after == 0:
            self.stop_price = None
            if fill.reason is OrderReason.STOP_LOSS:
                self._awaiting_flat_signal = True
        elif position_before == 0:
            self.stop_price = self._initial_stop(fill.price)

    def _trade_count(self, day: date) -> int:
        if day != self._trade_day:
            self._trade_day = day
            self._trades_today = 0
        return self._trades_today

    def _initial_stop(self, entry_price: Decimal) -> Decimal | None:
        candidates: list[Decimal] = []
        if self.config.stop_loss_fraction is not None:
            candidates.append(entry_price * (1 - self.config.stop_loss_fraction))
        if self.config.stop_loss_atr_multiple is not None and self._atr_value is not None:
            distance = Decimal(str(self.config.stop_loss_atr_multiple * self._atr_value))
            candidates.append(entry_price - distance)
        positive = [price for price in candidates if price > 0]
        return max(positive) if positive else None


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _optional_date(value: object) -> date | None:
    return None if value is None else date.fromisoformat(str(value))
