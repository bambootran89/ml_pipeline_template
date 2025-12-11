"""
Cross-validation orchestrator using CVInitializer, FoldRunner and CVAggregator.

This class is intentionally small: orchestration only, all heavy work is
delegated to helper classes to keep functions short and testable.
"""

from typing import Any, Dict, List, Tuple

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
    """Orchestrate cross-validation for time-series experiments."""

    def __init__(
        self,
        cfg: DictConfig,
        splitter: TimeSeriesSplitter,
        mlflow_manager: MLflowManager,
    ) -> None:
        """Initialize the CV pipeline."""
        super().__init__(cfg)
        self.splitter = splitter
        self.mlflow_manager = mlflow_manager
        self._printer = CVPrinter()
        self._aggregator = CVAggregator()
        self._initializer = CVInitializer(cfg, splitter)
        self._fold_runner = FoldRunner(cfg, mlflow_manager)

    def preprocess(self) -> Any:
        """Delegate preprocessing to the initializer (keeps API)."""
        # Reuse initializer to call DataModule preprocessing
        # but keep method for backward compatibility.
        # This returns the preprocessed dataset object.

        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def run_approach(self, approach: Dict[str, Any], data: Any) -> Any:
        """Not supported for CV pipeline; use run_cv()."""
        raise NotImplementedError("Use run_cv() for cross-validation")

    def _initialize_context(
        self, approach: Dict[str, Any], data: Any
    ) -> Tuple[Any, Any, str, Dict[str, Any], int]:
        """Small wrapper to expose initializer output without many locals."""
        return self._initializer.initialize(approach, data)

    def run_cv(
        self,
        approach: Dict[str, Any],
        data: Any,
        is_tuning: bool = False,  # <--- NEW PARAM
    ) -> Dict[str, float]:
        """
        Execute cross-validation for a given model approach and dataset.

        Args:
            approach (Dict[str, Any]):
                Dictionary specifying the model and hyperparameters.
                Example: {"model": "TFT", "hyperparams": {...}}
            data (Any):
                Preprocessed dataset used for training and validation.
            is_tuning (bool, optional):
                If True, disables heavy logging and MLflow CV summary.
                Used when called from a hyperparameter tuner. Defaults to False.

        Returns:
            Dict[str, float]:
                Aggregated metrics across all CV folds,
                including mean, std, min, and max
                for each metric.
        """
        x_full, y_full, model_name, hyperparams, _ = self._initialize_context(
            approach, data
        )

        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.splitter.split(x_full, y_full)
        ):
            fold_num = fold_idx + 1

            metrics = self._fold_runner.run_fold(
                fold_num,
                train_idx,
                test_idx,
                x_full,
                y_full,
                model_name,
                hyperparams,
                is_tuning=is_tuning,  # <--- Pass flag down
            )
            fold_metrics.append(metrics)

        aggregated = self._aggregator.aggregate(fold_metrics)
        self._printer.summary(aggregated)

        # If we are NOT tuning, we might want a CV summary run.
        # If we ARE tuning, the Tuner handles the logging of aggregated metrics.
        if (
            not is_tuning
            and self.mlflow_manager
            and getattr(self.mlflow_manager, "enabled", False)
        ):
            with self.mlflow_manager.start_run(run_name=f"{model_name}_cv_summary"):
                self.mlflow_manager.log_metrics(aggregated)

        return aggregated
