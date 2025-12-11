"""
Cross-validation orchestrator for time-series experiments.

This class handles orchestration only. Heavy computation is delegated
to CVInitializer, FoldRunner, and CVAggregator to keep functions short
and testable.
"""

from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from mlproject.src.datamodule.cv_data_prep import CVInitializer
from mlproject.src.datamodule.splitter import TimeSeriesSplitter
from mlproject.src.eval.aggregator import CVAggregator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.engines.cv_fold_runner import FoldRunner
from mlproject.src.pipeline.utils.printers import CVPrinter
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

    # def _initialize_context(
    #     self, approach: Dict[str, Any], data: Any = None
    # ) -> Tuple[Any, Any, str, Dict[str, Any], int]:
    #     """
    #     Prepare raw data and model info for CV.

    #     Args:
    #         approach: Model configuration dictionary.
    #         data: Not used (kept for API compatibility).

    #     Returns:
    #         x_full_raw: Raw input windows.
    #         y_full: Target windows.
    #         model_name: Model name string.
    #         hyperparams: Model hyperparameters.
    #         total_folds: Number of CV folds.
    #     """
    #     return self._initializer.initialize(approach)

    def run_cv(
        self, approach: Dict[str, Any], data: Any = None, is_tuning=False
    ) -> Dict[str, float]:
        """
        Execute cross-validation using raw input windows.
        """
        _ = data
        x_full_raw, y_full, model_name, hyperparams, _ = self._initializer.initialize(
            approach
        )

        fold_metrics: list[Dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.splitter.split(x_full_raw, y_full)
        ):
            # Convert np.ndarray to list[int] to satisfy type hints
            train_idx_list = (
                train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)
            )
            test_idx_list = (
                test_idx.tolist() if hasattr(test_idx, "tolist") else list(test_idx)
            )

            metrics = self._run_single_fold(
                fold_idx,
                train_idx_list,
                test_idx_list,
                x_full_raw,
                y_full,
                model_name,
                hyperparams,
                is_tuning,
            )
            fold_metrics.append(metrics)

        return self._aggregate_fold_metrics(fold_metrics)

    def _run_single_fold(
        self,
        fold_idx: int,
        train_idx: list[int],
        test_idx: list[int],
        x_full_raw: Any,
        y_full: Any,
        model_name: str,
        hyperparams: dict[str, Any],
        is_tuning: bool,
    ) -> dict[str, float]:
        """Run a single fold and return metrics dict."""
        fold_num = fold_idx + 1
        return self._fold_runner.run_fold(
            fold_num,
            train_idx,
            test_idx,
            x_full_raw,
            y_full,
            model_name,
            hyperparams,
            is_tuning=is_tuning,
        )

    def _aggregate_fold_metrics(
        self, fold_metrics: list[dict[str, float]]
    ) -> dict[str, float]:
        """Compute mean and std for each metric across folds."""
        if not fold_metrics:
            return {}

        agg: dict[str, float] = {}
        for key in fold_metrics[0].keys():
            values = [f[key] for f in fold_metrics]
            agg[f"{key}_mean"] = float(sum(values) / len(values))
            agg[f"{key}_std"] = float(np.std(values))
        return agg
