"""
Aggregation utilities for cross-validation metrics.

Offers mean/std/min/max aggregation across folds.
"""

from typing import Dict, List, Set

import numpy as np


class CVAggregator:
    """Aggregate per-fold metrics into summary statistics."""

    @staticmethod
    def aggregate(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate fold metrics.

        Args:
            fold_metrics: List of per-fold metric dicts.

        Returns:
            Dictionary containing mean/std/min/max for each metric.
        """
        if not fold_metrics:
            return {}

        all_keys: Set[str] = set()  # type annotation added
        for m in fold_metrics:
            all_keys.update(m.keys())

        result: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [m[key] for m in fold_metrics if key in m]
            arr = np.asarray(values, dtype=float)
            result[f"{key}_mean"] = float(np.mean(arr))
            result[f"{key}_std"] = float(np.std(arr))
            result[f"{key}_min"] = float(np.min(arr))
            result[f"{key}_max"] = float(np.max(arr))

        return result
