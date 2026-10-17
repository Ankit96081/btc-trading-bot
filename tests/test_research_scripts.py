"""End-to-end tests for the dataset-building and backtest scripts (no network)."""

import csv
from datetime import timedelta
from pathlib import Path

import pytest

from scripts import build_dataset, run_backtest
from src.backtest.cli import write_candles_csv
from src.data.dataset import load_candles
from tests.factories import DEFAULT_START, bars_from_closes


def write_minute_data(directory: Path) -> Path:
    closes = [100 + (index % 7) - (index % 3) for index in range(8 * 60)]
    candles = bars_from_closes(closes, timeframe="1m", spread="0.5")
    # Remove one minute to create an exchange-style gap in the third hour.
    gap_minute = DEFAULT_START + timedelta(hours=2, minutes=30)
    candles = [candle for candle in candles if candle.open_time != gap_minute]
    path = directory / "btcusdt_1m_test.csv"
    write_candles_csv(path, candles)
    return path


def test_build_dataset_resamples_and_reports_gaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = write_minute_data(tmp_path)
    output_dir = tmp_path / "processed"

    exit_code = build_dataset.main(
        ["--source", str(source), "--timeframes", "1h", "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    assert "Data gaps: 1" in capsys.readouterr().out
    hourly = load_candles(output_dir / "btcusdt_1h.csv", allow_gaps=True)
    assert len(hourly) == 7
    assert DEFAULT_START + timedelta(hours=2) not in {bar.open_time for bar in hourly}


def test_build_dataset_fails_cleanly_without_source_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = build_dataset.main(["--source", str(tmp_path / "missing_*.csv")])

    assert exit_code == 1
    assert "Dataset build failed" in capsys.readouterr().err


def test_run_backtest_prints_metrics_and_writes_fills(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "btcusdt_1h.csv"
    write_candles_csv(data, bars_from_closes([100, 102, 101, 105, 107, 104, 110, 111], spread="1"))
    fills = tmp_path / "fills.csv"

    exit_code = run_backtest.main(
        [
            "--data",
            str(data),
            "--strategy",
            "ma_crossover",
            "--param",
            "fast_period=2",
            "--param",
            "slow_period=3",
            "--stop-pct",
            "0.05",
            "--fills-output",
            str(fills),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Sharpe ratio" in output
    assert "Buy & hold return" in output
    with fills.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    assert rows and rows[0]["side"] == "BUY"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--strategy", "unknown"], "unknown strategy"),
        (["--param", "window=3", "--strategy", "ma_crossover"], "no parameter"),
        (["--max-drawdown", "1.5"], "between 0 and 1"),
    ],
)
def test_run_backtest_reports_configuration_errors(
    tmp_path: Path, extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "btcusdt_1h.csv"
    write_candles_csv(data, bars_from_closes([100, 101, 102]))

    exit_code = run_backtest.main(["--data", str(data), *extra])

    assert exit_code == 1
    assert message in capsys.readouterr().err


def test_walk_forward_script_reports_verdict_and_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import walk_forward
    from tests.test_walk_forward import trending_market

    data = tmp_path / "btcusdt_1h.csv"
    bars = trending_market(bars=24 * 60)
    write_candles_csv(data, bars)
    holdout_start = bars[24 * 50].open_time.isoformat().replace("+00:00", "Z")
    report = tmp_path / "report.json"

    exit_code = walk_forward.main(
        [
            "--data", str(data),
            "--grid", "fast_period=5,10",
            "--grid", "slow_period=30,60",
            "--train-days", "20",
            "--test-days", "10",
            "--warmup-bars", "100",
            "--min-train-trips", "1",
            "--holdout-start", holdout_start,
            "--evaluate-holdout",
            "--report", str(report),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "VERDICT:" in output
    assert "HOLDOUT VERDICT:" in output
    assert '"criteria"' in report.read_text(encoding="utf-8")


def test_walk_forward_script_rejects_conflicting_parameters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import walk_forward

    exit_code = walk_forward.main(
        ["--grid", "fast_period=5,10", "--param", "fast_period=5", "--data", str(tmp_path)]
    )

    assert exit_code == 1
    assert "both as --param and --grid" in capsys.readouterr().err
