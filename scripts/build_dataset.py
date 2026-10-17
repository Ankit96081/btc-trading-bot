"""Combine raw 1m candle files and resample them into research datasets.

Example:
    python scripts/build_dataset.py --timeframes 15m 1h 4h

Reads data/raw/btcusdt_1m_*.csv, reports exchange data gaps, and writes one
complete-bars-only CSV per timeframe to data/processed/btcusdt_<timeframe>.csv.
Bars touched by a gap are dropped, never filled in.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.cli import resolve_paths, write_candles_csv
from src.data.dataset import DatasetError, find_gaps, load_candles
from src.data.resample import ResampleError, resample_candles

DEFAULT_SOURCE = "data/raw/btcusdt_1m_*.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=[DEFAULT_SOURCE], metavar="PATH_OR_GLOB")
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(arguments)

    try:
        paths = resolve_paths(args.source, PROJECT_ROOT)
        print(f"Loading {len(paths)} file(s)...")
        candles = load_candles(paths, allow_gaps=True)
        gaps = find_gaps(candles)
        print(f"Loaded {len(candles):,} {candles[0].timeframe} candles from "
              f"{candles[0].open_time:%Y-%m-%d %H:%M} to {candles[-1].open_time:%Y-%m-%d %H:%M} UTC")
        print(f"Data gaps: {len(gaps)}")
        for before, after in gaps:
            print(f"  {before.open_time:%Y-%m-%d %H:%M} -> {after.open_time:%Y-%m-%d %H:%M} UTC")

        for timeframe in args.timeframes:
            bars = resample_candles(candles, timeframe)
            symbol = candles[0].symbol.replace("/", "").lower()
            output = args.output_dir / f"{symbol}_{timeframe}.csv"
            write_candles_csv(output, bars)
            print(f"Wrote {len(bars):,} {timeframe} bars to {output}")
    except (FileNotFoundError, DatasetError, ResampleError, OSError) as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
