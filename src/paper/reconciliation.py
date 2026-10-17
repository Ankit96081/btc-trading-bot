"""Align the engine's view of the account with the exchange after a (re)start."""

import logging
from decimal import Decimal

from src.core.engine import TradingEngine
from src.execution.binance_testnet import TestnetBroker

LOGGER = logging.getLogger(__name__)


def reconcile_with_exchange(engine: TradingEngine, broker: TestnetBroker) -> list[str]:
    """Recover in-flight orders and correct the position from exchange balances.

    The testnet account is shared and pre-funded, so the bot's position is
    measured against the base-asset balance recorded when the run was
    created.  The exchange is the source of truth: a mismatch is corrected
    and reported, and an impossible state (less base asset than the baseline,
    for example after a testnet reset) halts trading.  Returns warnings.
    """

    warnings: list[str] = []
    recovered = broker.recover_in_flight_order()
    if recovered:
        engine.apply_fills(recovered)
        warnings.append(f"recovered {len(recovered)} fill(s) from an order sent before a restart")

    balances = broker.client.balances()
    base = balances.get(broker.rules.base_asset, Decimal(0))
    quote = balances.get(broker.rules.quote_asset, Decimal(0))
    if broker.baseline_base_balance is None:
        broker.baseline_base_balance = base - engine.portfolio.quantity

    exchange_position = base - broker.baseline_base_balance
    if exchange_position < 0:
        message = (
            f"exchange {broker.rules.base_asset} balance {base} is below the run's baseline "
            f"{broker.baseline_base_balance}; the account changed outside this bot"
        )
        engine.risk.halt(message)
        warnings.append(message)
    elif abs(exchange_position - engine.portfolio.quantity) >= broker.rules.step_size:
        warnings.append(
            f"position mismatch: engine {engine.portfolio.quantity}, exchange {exchange_position}; "
            "using the exchange value"
        )
        engine.portfolio.quantity = broker.rules.round_quantity(exchange_position)

    if quote < engine.portfolio.cash:
        warnings.append(
            f"exchange {broker.rules.quote_asset} balance {quote} is below the run's cash "
            f"{engine.portfolio.cash}; limiting cash to the exchange balance"
        )
        engine.portfolio.cash = quote

    for warning in warnings:
        LOGGER.warning(warning)
    return warnings
