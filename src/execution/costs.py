"""Trading-cost assumptions applied to every simulated fill."""

from dataclasses import dataclass
from decimal import Decimal

from .models import Side

_BPS = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Proportional fees and slippage.

    Defaults are deliberately conservative: Binance Spot's standard 0.1% taker
    fee (market orders are always takers) plus 2 bps of adverse slippage on
    BTC/USDT, which is very liquid at the sizes this project targets.
    """

    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal(2)

    def __post_init__(self) -> None:
        if self.fee_rate < 0 or self.slippage_bps < 0:
            raise ValueError("fee_rate and slippage_bps must not be negative")

    def execution_price(self, side: Side, reference_price: Decimal) -> Decimal:
        """Return ``reference_price`` moved against the trader by the slippage."""

        slippage = reference_price * self.slippage_bps * _BPS
        return reference_price + slippage if side is Side.BUY else reference_price - slippage

    def fee(self, notional: Decimal) -> Decimal:
        return notional * self.fee_rate

    def scaled(self, factor: Decimal | int) -> "CostModel":
        """Return a model with fees and slippage multiplied by ``factor`` (for stress tests)."""

        multiplier = Decimal(factor)
        return CostModel(
            fee_rate=self.fee_rate * multiplier, slippage_bps=self.slippage_bps * multiplier
        )
