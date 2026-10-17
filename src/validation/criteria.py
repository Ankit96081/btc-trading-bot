"""Predefined pass/fail criteria; see docs/validation_criteria.md.

Thresholds here must match that document.  Do not tune them to make a
strategy pass.
"""

from dataclasses import dataclass

from src.backtest.metrics import PerformanceMetrics


@dataclass(frozen=True, slots=True)
class CriterionResult:
    name: str
    passed: bool
    actual: str
    threshold: str


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """The walk-forward outputs the criteria are judged on."""

    out_of_sample: PerformanceMetrics
    profitable_fold_fraction: float
    stressed_sharpe_ratio: float
    median_neighbor_ratio: float | None


@dataclass(frozen=True, slots=True)
class ValidationCriteria:
    min_total_return: float = 0.0
    min_sharpe_ratio: float = 0.8
    max_drawdown: float = 0.25
    min_profitable_fold_fraction: float = 0.6
    min_round_trips: int = 30
    max_fills_per_day: float = 10.0
    min_stressed_sharpe_ratio: float = 0.5
    min_neighbor_ratio: float = 0.5

    def evaluate(self, evidence: ValidationEvidence) -> list[CriterionResult]:
        metrics = evidence.out_of_sample
        neighbor = evidence.median_neighbor_ratio
        return [
            CriterionResult(
                "OOS total return",
                metrics.total_return > self.min_total_return,
                _pct(metrics.total_return),
                f"> {_pct(self.min_total_return)}",
            ),
            CriterionResult(
                "OOS Sharpe ratio",
                metrics.sharpe_ratio >= self.min_sharpe_ratio,
                f"{metrics.sharpe_ratio:.2f}",
                f">= {self.min_sharpe_ratio:.2f}",
            ),
            CriterionResult(
                "OOS Sharpe vs buy-and-hold",
                metrics.sharpe_ratio >= metrics.benchmark_sharpe_ratio,
                f"{metrics.sharpe_ratio:.2f}",
                f">= {metrics.benchmark_sharpe_ratio:.2f}",
            ),
            CriterionResult(
                "OOS max drawdown",
                metrics.max_drawdown <= self.max_drawdown,
                _pct(metrics.max_drawdown),
                f"<= {_pct(self.max_drawdown)}",
            ),
            CriterionResult(
                "Profitable folds",
                evidence.profitable_fold_fraction >= self.min_profitable_fold_fraction,
                _pct(evidence.profitable_fold_fraction),
                f">= {_pct(self.min_profitable_fold_fraction)}",
            ),
            CriterionResult(
                "OOS round trips",
                metrics.round_trips >= self.min_round_trips,
                str(metrics.round_trips),
                f">= {self.min_round_trips}",
            ),
            CriterionResult(
                "Fills per day",
                metrics.fills_per_day <= self.max_fills_per_day,
                f"{metrics.fills_per_day:.2f}",
                f"<= {self.max_fills_per_day:.1f}",
            ),
            CriterionResult(
                "Sharpe with 2x costs",
                evidence.stressed_sharpe_ratio >= self.min_stressed_sharpe_ratio,
                f"{evidence.stressed_sharpe_ratio:.2f}",
                f">= {self.min_stressed_sharpe_ratio:.2f}",
            ),
            CriterionResult(
                "Parameter robustness",
                neighbor is not None and neighbor >= self.min_neighbor_ratio,
                "n/a (no neighbors)" if neighbor is None else f"{neighbor:.2f}",
                f">= {self.min_neighbor_ratio:.2f}",
            ),
        ]


@dataclass(frozen=True, slots=True)
class HoldoutCriteria:
    min_total_return: float = 0.0
    max_drawdown: float = 0.25

    def evaluate(self, metrics: PerformanceMetrics) -> list[CriterionResult]:
        return [
            CriterionResult(
                "Holdout total return",
                metrics.total_return > self.min_total_return,
                _pct(metrics.total_return),
                f"> {_pct(self.min_total_return)}",
            ),
            CriterionResult(
                "Holdout max drawdown",
                metrics.max_drawdown <= self.max_drawdown,
                _pct(metrics.max_drawdown),
                f"<= {_pct(self.max_drawdown)}",
            ),
        ]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
