"""Deterministic candle builders shared by the test suite."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.data.models import OHLCVCandle
from src.data.timeframes import timeframe_to_timedelta

DEFAULT_START = datetime(2024, 1, 1, tzinfo=UTC)


def make_bar(
    open_time: datetime,
    *,
    open: Decimal | float | str,
    high: Decimal | float | str,
    low: Decimal | float | str,
    close: Decimal | float | str,
    volume: Decimal | float | str = "1",
    timeframe: str = "1m",
    symbol: str = "BTC/USDT",
) -> OHLCVCandle:
    return OHLCVCandle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        close_time=open_time + timeframe_to_timedelta(timeframe) - timedelta(milliseconds=1),
        open=Decimal(str(open)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
    )


def bars_from_closes(
    closes: Sequence[float | str | Decimal],
    *,
    timeframe: str = "1h",
    start: datetime = DEFAULT_START,
    spread: str = "0",
) -> list[OHLCVCandle]:
    """Build contiguous bars whose open is the previous close.

    ``spread`` widens each bar's high/low around its open/close range.
    """

    step = timeframe_to_timedelta(timeframe)
    margin = Decimal(spread)
    bars: list[OHLCVCandle] = []
    previous_close = Decimal(str(closes[0]))
    for index, value in enumerate(closes):
        close = Decimal(str(value))
        open_price = previous_close
        bars.append(
            make_bar(
                start + index * step,
                open=open_price,
                high=max(open_price, close) + margin,
                low=min(open_price, close) - margin,
                close=close,
                timeframe=timeframe,
            )
        )
        previous_close = close
    return bars
