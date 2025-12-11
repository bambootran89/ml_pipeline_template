"""
Aggregate and visualize metrics from cross-validation folds.

Features:
- Compute mean ± std for each metric.
- Generate summary report.
- Identify best-performing fold.
"""

from typing import Dict, List

import numpy as np


class CVEvaluator:
    """Evaluator class for aggregating and inspecting CV metrics.

    Example:
        >>> evaluator = CVEvaluator()
        >>> fold_metrics = [
        ...     {"mae": 0.5, "mse": 0.3},
        ...     {"mae": 0.6, "mse": 0.35},
        ...     {"mae": 0.55, "mse": 0.32}
        ... ]
        >>> summary = evaluator.aggregate(fold_metrics)
        >>> print(summary)
        {'mae_mean': 0.55, 'mae_std': 0.041..., ...}
    """

    def __init__(self) -> None:
        """Initialize the cross-validation evaluator."""

    def aggregate(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across folds.

        Args:
            fold_metrics (List[Dict[str, float]]): A list where each item is a
                dictionary containing metrics of a single fold.

        Returns:
            Dict[str, float]: Aggregated metrics (mean, std, min, max).
        """
        if not fold_metrics:
            return {}

        # Collect all metric keys.
        all_keys: set[str] = set()  # <<< FIX
        for metrics in fold_metrics:
            all_keys.update(metrics.keys())

        # Compute aggregated statistics.
        result: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [m[key] for m in fold_metrics if key in m]
            if values:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_std"] = float(np.std(values))
                result[f"{key}_min"] = float(np.min(values))
                result[f"{key}_max"] = float(np.max(values))

        return result

    def print_summary(
        self, fold_metrics: List[Dict[str, float]], model_name: str = ""
    ) -> None:
        """Print a formatted summary report of aggregated CV metrics."""
        aggregated = self.aggregate(fold_metrics)

        print(f"\n{'=' * 70}")
        if model_name:
            title = f"CROSS-VALIDATION SUMMARY - {model_name.upper()}"
        else:
            title = "CROSS-VALIDATION SUMMARY"
        print(f"  {title}")
        print(f"{'=' * 70}")
        print(f"Number of folds: {len(fold_metrics)}")
        print(f"{'-' * 70}")

        # Identify base metric names.
        metric_names = set()
        for key in aggregated:
            if key.endswith("_mean"):
                metric_names.add(key.replace("_mean", ""))

        # Print each aggregated metric.
        for metric in sorted(metric_names):
            mean = aggregated.get(f"{metric}_mean", 0.0)
            std = aggregated.get(f"{metric}_std", 0.0)
            min_val = aggregated.get(f"{metric}_min", 0.0)
            max_val = aggregated.get(f"{metric}_max", 0.0)

            print(
                f"{metric.upper():12} | "
                f"Mean: {mean:10.6f} ± {std:10.6f} | "
                f"Range: [{min_val:10.6f}, {max_val:10.6f}]"
            )

        print(f"{'=' * 70}\n")

    def get_best_fold(
        self,
        fold_metrics: List[Dict[str, float]],
        metric_name: str = "mae",
        minimize: bool = True,
    ) -> int:
        """Return the index of the best-performing fold."""
        values = [m[metric_name] for m in fold_metrics if metric_name in m]
        if not values:
            raise ValueError(f"Metric '{metric_name}' not found in fold_metrics")

        if minimize:
            return int(np.argmin(values))
        return int(np.argmax(values))
