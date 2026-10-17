"""Exact cash-and-position accounting for a single spot market."""

from decimal import Decimal

from src.execution.models import Fill, Side


class InsufficientFundsError(RuntimeError):
    """Raised when a fill would make cash or the position negative."""


class Portfolio:
    """Quote-currency cash plus a long-only base-asset position."""

    def __init__(self, cash: Decimal, quantity: Decimal = Decimal(0)) -> None:
        if cash < 0 or quantity < 0:
            raise ValueError("cash and quantity must not be negative")
        self.cash = cash
        self.quantity = quantity

    def apply(self, fill: Fill) -> None:
        if fill.side is Side.BUY:
            cost = fill.notional + fill.fee
            if cost > self.cash:
                raise InsufficientFundsError(
                    f"buy costs {cost} but only {self.cash} cash is available"
                )
            self.cash -= cost
            self.quantity += fill.quantity
        else:
            if fill.quantity > self.quantity:
                raise InsufficientFundsError(
                    f"cannot sell {fill.quantity}; position is {self.quantity}"
                )
            self.cash += fill.notional - fill.fee
            self.quantity -= fill.quantity

    def equity(self, price: Decimal) -> Decimal:
        return self.cash + self.quantity * price

    def exposure(self, price: Decimal) -> Decimal:
        equity = self.equity(price)
        if equity <= 0:
            return Decimal(0)
        return self.quantity * price / equity
