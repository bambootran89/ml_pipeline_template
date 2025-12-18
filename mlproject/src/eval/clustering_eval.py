from typing import Any, Dict

import numpy as np
from sklearn.metrics import silhouette_score

from mlproject.src.eval.base import BaseEvaluator


class ClusteringEvaluator(BaseEvaluator):
    """
    Evaluator for clustering models.

    This evaluator supports intrinsic clustering metrics that do not
    require ground-truth labels, including:
    - inertia (within-cluster sum of squares)
    - silhouette_score
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute clustering metrics.

        Args:
            y_true:
                Feature matrix X.
            y_pred:
                Cluster labels.
            **kwargs:
                Optional extra information:
                - model: fitted clustering model (to extract inertia)

        Returns:
            Dict[str, float]:
                Clustering metrics.
        """
        x = kwargs.get("x")
        labels = np.asarray(y_pred)

        metrics: Dict[str, float] = {}

        model = kwargs.get("model")

        if model is not None and hasattr(model, "inertia_"):
            metrics["inertia"] = float(model.inertia_)

        n_clusters = np.unique(labels).size

        if x is not None and x.ndim == 2 and n_clusters > 1:
            metrics["silhouette_score"] = float(silhouette_score(x, labels))
        return metrics
