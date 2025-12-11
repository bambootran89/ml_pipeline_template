"""
Cross-validation orchestrator for time-series experiments.

This class handles orchestration only. Heavy computation is delegated
to CVInitializer, FoldRunner, and CVAggregator to keep functions short
and testable.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from mlproject.src.cv.cv_aggregator import CVAggregator
from mlproject.src.cv.cv_initializer import CVInitializer
from mlproject.src.cv.fold_runner import FoldRunner
from mlproject.src.cv.printers import CVPrinter
from mlproject.src.cv.splitter import TimeSeriesSplitter
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager


class CrossValidationPipeline(BasePipeline):
    """
    Orchestrate cross-validation using raw data windows and fold runners.
    """

    def __init__(
        self,
        cfg: DictConfig,
        splitter: TimeSeriesSplitter,
        mlflow_manager: MLflowManager,
    ) -> None:
        """
        Initialize the cross-validation pipeline.

        Args:
            cfg: Configuration object.
            splitter: TimeSeriesSplitter defining CV folds.
            mlflow_manager: MLflow manager for logging.
        """
        super().__init__(cfg)
        self.splitter = splitter
        self.mlflow_manager = mlflow_manager
        self._printer = CVPrinter()
        self._aggregator = CVAggregator()
        self._initializer = CVInitializer(cfg, splitter)
        self._fold_runner = FoldRunner(cfg, mlflow_manager)

    def preprocess(self) -> Any:
        """
        Delegate preprocessing to the OfflinePreprocessor.

        Returns:
            Preprocessed dataset (for API compatibility).
        """
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def run_approach(self, approach: Dict[str, Any], data: Any) -> Any:
        """
        Not supported for CV pipeline.

        Raises:
            NotImplementedError: Cross-validation uses run_cv().
        """
        raise NotImplementedError("Use run_cv() for cross-validation")

    def _initialize_context(
        self, approach: Dict[str, Any], data: Any = None
    ) -> Tuple[Any, Any, str, Dict[str, Any], int]:
        """
        Prepare raw data and model info for CV.

        Args:
            approach: Model configuration dictionary.
            data: Not used (kept for API compatibility).

        Returns:
            x_full_raw: Raw input windows.
            y_full: Target windows.
            model_name: Model name string.
            hyperparams: Model hyperparameters.
            total_folds: Number of CV folds.
        """
        return self._initializer.initialize(approach)

    def run_cv(
        self, approach: Dict[str, Any], data: Any = None, is_tuning=False
    ) -> Dict[str, Any]:
        """
        Execute cross-validation using raw input windows.

        Args:
            approach: Model configuration.
            data: Not used (data is loaded by initializer).
            is_tuning: Flag indicating if hyperparameter tuning is active.

        Returns:
            List of metrics dictionaries, one per fold.
        """
        x_full_raw, y_full, model_name, hyperparams, _ = self._initializer.initialize(
            approach
        )

        fold_metrics: List[Dict[str, Any]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.splitter.split(x_full_raw, y_full)
        ):
            fold_num = fold_idx + 1
            metrics = self._fold_runner.run_fold(
                fold_num,
                train_idx,
                test_idx,
                x_full_raw,  # Raw data
                y_full,
                model_name,
                hyperparams,
                is_tuning=is_tuning,
            )
            fold_metrics.append(metrics)

        metrics_agg = {}
        for key in fold_metrics[0].keys():
            values = [f[key] for f in fold_metrics]
            metrics_agg[f"{key}_mean"] = sum(values) / len(values)
            metrics_agg[f"{key}_std"] = np.std(values)

        return metrics_agg
