"""Walk-forward validate a strategy against the predefined criteria.

Example:
    python scripts/walk_forward.py --strategy ma_crossover \
        --grid fast_period=12,24,48 --grid slow_period=96,168,336 --stop-atr 3

Grid or --param names starting with "risk." tune risk settings per fold, e.g.
    --grid risk.vol_target=0.3,0.4,0.6
(choices: risk.vol_target, risk.vol_lookback, risk.stop_atr).

Data from --holdout-start onwards is excluded from tuning.  Evaluate it with
--evaluate-holdout only once a strategy passes every criterion; see
docs/validation_criteria.md.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.cli import (
    add_cost_arguments,
    add_data_arguments,
    add_risk_arguments,
    add_strategy_arguments,
    cost_model_from,
    engine_config_from,
    format_metrics,
    load_bars,
    parse_key_values,
    parse_utc_timestamp,
    resolve_paths,
    risk_config_from,
)
from src.backtest.runner import run_backtest
from src.data.dataset import DatasetError
from src.data.timeframes import timeframe_to_minutes
from src.strategies.registry import StrategyConfigurationError, strategy_factory
from src.validation import (
    CriterionResult,
    HoldoutCriteria,
    ValidationCriteria,
    ValidationEvidence,
    WalkForwardConfig,
    WalkForwardResult,
    expand_grid,
    fold_parameters,
    run_walk_forward,
    split_parameters,
)

DEFAULT_DATA = "data/processed/btcusdt_1h.csv"
DEFAULT_HOLDOUT_START = datetime(2026, 3, 1, tzinfo=UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_arguments(parser, DEFAULT_DATA)
    add_strategy_arguments(parser, "ma_crossover")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="NAME=V1,V2,...",
        help="Values to search for one parameter; repeat per parameter.",
    )
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--warmup-bars", type=int, default=500)
    parser.add_argument("--min-train-trips", type=int, default=5)
    parser.add_argument("--anchored", action="store_true", help="Expanding train windows.")
    parser.add_argument(
        "--holdout-start",
        type=parse_utc_timestamp,
        default=DEFAULT_HOLDOUT_START,
        help="First timestamp of the untouched holdout (default: 2026-03-01T00:00:00Z).",
    )
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Also test the final parameters on the holdout. Do this once per strategy.",
    )
    parser.add_argument("--report", type=Path, help="Write a JSON report to this path.")
    add_cost_arguments(parser)
    add_risk_arguments(parser)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        grid = {name: values.split(",") for name, values in parse_key_values(args.grid).items()}
        fixed = parse_key_values(args.param)
        overlap = set(grid) & set(fixed)
        if overlap:
            raise ValueError(f"parameters given both as --param and --grid: {sorted(overlap)}")
        grid.update({name: [value] for name, value in fixed.items()})
        for parameters in expand_grid(grid)[:1]:  # Fail fast on unknown names.
            strategy_parameters, _ = split_parameters(parameters, None)
            strategy_factory(args.strategy, strategy_parameters)

        all_bars = load_bars(resolve_paths(args.data, PROJECT_ROOT), args.start, args.end)
        research = [bar for bar in all_bars if bar.open_time < args.holdout_start]
        holdout = [bar for bar in all_bars if bar.open_time >= args.holdout_start]
        if not research:
            raise DatasetError("no research data before the holdout start")
        bars_per_day = 1_440 // timeframe_to_minutes(research[0].timeframe)
        config = WalkForwardConfig(
            train_bars=args.train_days * bars_per_day,
            test_bars=args.test_days * bars_per_day,
            anchored=args.anchored,
            warmup_bars=args.warmup_bars,
            min_train_round_trips=args.min_train_trips,
        )
        common: dict[str, Any] = {
            "initial_cash": args.initial_cash,
            "risk_config": risk_config_from(args),
            "engine_config": engine_config_from(args),
        }
        cost_model = cost_model_from(args)
        result = run_walk_forward(
            research, args.strategy, grid, config, cost_model=cost_model, **common
        )
        stressed = run_walk_forward(
            research,
            args.strategy,
            grid,
            config,
            cost_model=cost_model.scaled(2),
            fixed_parameters=fold_parameters(result),
            **common,
        )
    except (
        argparse.ArgumentTypeError,
        DatasetError,
        FileNotFoundError,
        StrategyConfigurationError,
        ValueError,
    ) as error:
        print(f"Walk-forward failed: {error}", file=sys.stderr)
        return 1

    print(
        f"{args.strategy}: {len(result.folds)} folds of {args.test_days} days after "
        f"{args.train_days}-day {'anchored' if args.anchored else 'rolling'} training"
    )
    print(f"{'Fold':<5} {'Test period':<25} {'Parameters':<40} {'Train Sharpe':>12} "
          f"{'Test return':>11} {'Test Sharpe':>11}")
    for fold in result.folds:
        metrics = fold.test_metrics
        period = f"{fold.test_start.open_time:%Y-%m-%d}..{fold.test_end.open_time:%Y-%m-%d}"
        parameters = ", ".join(f"{key}={value}" for key, value in fold.parameters.items())
        print(f"{fold.index + 1:<5} {period:<25} {parameters:<40} {fold.train_score:>12.2f} "
              f"{metrics.total_return * 100:>10.2f}% {metrics.sharpe_ratio:>11.2f}")

    print("\nStitched out-of-sample performance:")
    print(format_metrics(result.out_of_sample))

    evidence = ValidationEvidence(
        out_of_sample=result.out_of_sample,
        profitable_fold_fraction=result.profitable_fold_fraction,
        stressed_sharpe_ratio=stressed.out_of_sample.sharpe_ratio,
        median_neighbor_ratio=result.median_neighbor_ratio,
    )
    checks = ValidationCriteria().evaluate(evidence)
    passed = all(check.passed for check in checks)
    print("\nValidation criteria (docs/validation_criteria.md):")
    _print_checks(checks)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")

    holdout_checks: list[CriterionResult] = []
    if args.evaluate_holdout:
        holdout_checks = _evaluate_holdout(
            args, research, holdout, result.final_parameters, common, cost_model
        )
    if args.report:
        _write_report(args.report, args, result, checks, holdout_checks)
        print(f"Report written to {args.report}")
    return 0


def _evaluate_holdout(args, research, holdout, parameters, common, cost_model):
    if not holdout:
        print("\nHoldout: no data at or after the holdout start.")
        return []
    if len(holdout) < 2:
        print("\nHoldout: not enough data.")
        return []
    warmup = research[-args.warmup_bars:] if args.warmup_bars else []
    strategy_parameters, risk_config = split_parameters(parameters, common["risk_config"])
    holdout_result = run_backtest(
        [*warmup, *holdout],
        strategy_factory(args.strategy, strategy_parameters),
        cost_model=cost_model,
        warmup_bars=len(warmup),
        **{**common, "risk_config": risk_config},
    )
    print(f"\nHoldout ({holdout[0].open_time:%Y-%m-%d}..{holdout[-1].open_time:%Y-%m-%d}) "
          f"with final parameters {parameters}:")
    print(format_metrics(holdout_result.metrics))
    checks = HoldoutCriteria().evaluate(holdout_result.metrics)
    _print_checks(checks)
    verdict = "PASS" if all(check.passed for check in checks) else "FAIL"
    print(f"HOLDOUT VERDICT: {verdict}")
    return checks


def _print_checks(checks: Sequence[CriterionResult]) -> None:
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"  [{mark}] {check.name:<28} {check.actual:>12}  (needs {check.threshold})")


def _write_report(
    path: Path,
    args: argparse.Namespace,
    result: WalkForwardResult,
    checks: Sequence[CriterionResult],
    holdout_checks: Sequence[CriterionResult],
) -> None:
    report = {
        "strategy": args.strategy,
        "generated_at": datetime.now(UTC).isoformat(),
        "folds": [
            {
                "test_start": fold.test_start.open_time.isoformat(),
                "test_end": fold.test_end.open_time.isoformat(),
                "parameters": fold.parameters,
                "train_sharpe": fold.train_score,
                "neighbor_ratio": fold.neighbor_ratio,
                "test_metrics": fold.test_metrics.as_dict(),
            }
            for fold in result.folds
        ],
        "out_of_sample": result.out_of_sample.as_dict(),
        "criteria": [_check_dict(check) for check in checks],
        "holdout": [_check_dict(check) for check in holdout_checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _check_dict(check: CriterionResult) -> dict[str, object]:
    return {
        "name": check.name,
        "passed": check.passed,
        "actual": check.actual,
        "threshold": check.threshold,
    }


if __name__ == "__main__":
    raise SystemExit(main())
