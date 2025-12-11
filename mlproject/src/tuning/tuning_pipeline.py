#!/usr/bin/env python
"""
Tuning Pipeline: orchestrates hyperparameter tuning and final retraining.

Workflow:
    1. Run hyperparameter tuning (Optuna or Ray Tune).
    2. Retrieve best hyperparameters from the tuner.
    3. Retrain the model on the full dataset using best hyperparameters.
    4. Register the final model in MLflow Model Registry.
"""

from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig

from mlproject.src.cv.splitter import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
    TimeSeriesSplitter,
)
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.pipeline.training_pipeline import TrainingPipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna_tuner import OptunaTuner
from mlproject.src.tuning.ray_tuner import RayTuner


class TuningPipeline(BasePipeline):
    """
    End-to-end pipeline for hyperparameter tuning and full retraining.

    Example:
        >>> pipeline = TuningPipeline("configs/experiments/etth1.yaml")
        >>> best_params = pipeline.run()
    """

    def __init__(self, cfg_path: str = "") -> None:
        """
        Initialize the tuning pipeline.

        Args:
            cfg_path: Path to the experiment configuration YAML file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)

        # Use common base class type for splitter to satisfy MyPy
        self.splitter: TimeSeriesSplitter

        tuning_cfg = self.cfg.get("tuning", {})
        cv_strategy = tuning_cfg.get("cv_strategy", "expanding")
        n_splits = tuning_cfg.get("n_splits", 3)
        test_size = tuning_cfg.get("test_size", 20)

        if cv_strategy == "expanding":
            self.splitter = ExpandingWindowSplitter(n_splits, test_size)
        elif cv_strategy == "sliding":
            train_size = tuning_cfg.get("train_size", 60)
            self.splitter = SlidingWindowSplitter(n_splits, train_size, test_size)
        else:
            raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    def preprocess(self) -> Any:
        """
        Load and preprocess raw data before tuning.

        Returns:
            Any: Output from the offline preprocessing pipeline.
        """
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def run_approach(self, approach: Dict[str, Any], data: Any) -> None:
        """Not applicable in tuning pipeline."""
        raise NotImplementedError("Use run() for tuning workflow.")

    def _load_tuner(self, tuner_type: str) -> Union[OptunaTuner, RayTuner]:
        """
        Lazily load tuner class without violating import placement rules.

        Args:
            tuner_type: 'optuna' or 'ray'

        Returns:
            Instance of tuner.
        """
        metric_name = self.cfg.get("tuning", {}).get("optimize_metric", "mae_mean")
        direction = self.cfg.get("tuning", {}).get("direction", "minimize")

        if tuner_type == "optuna":
            return OptunaTuner(
                cfg=self.cfg,
                splitter=self.splitter,
                mlflow_manager=self.mlflow_manager,
                metric_name=metric_name,
                direction=direction,
            )

        if tuner_type == "ray":
            return RayTuner(
                cfg=self.cfg,
                splitter=self.splitter,
                mlflow_manager=self.mlflow_manager,
                metric_name=metric_name,
                direction=direction,
            )

        raise ValueError(f"Unknown tuner_type: {tuner_type}")

    def run(
        self, data: Optional[Any] = None, tuner_type: str = "optuna"
    ) -> Dict[str, Any]:
        """
        Execute the full hyperparameter tuning workflow.

        Args:
            data: Optional preprocessed dataset. If None, preprocessing is run.
            tuner_type: Backend tuner type. Either 'optuna' or 'ray'.

        Returns:
            dict: Best hyperparameters found during tuning.
        """
        if data is None:
            data = self.preprocess()

        tuner = self._load_tuner(tuner_type)

        tuning_cfg = self.cfg.get("tuning", {})
        n_trials = tuning_cfg.get("n_trials", 20)
        timeout = tuning_cfg.get("timeout")
        n_jobs = tuning_cfg.get("n_jobs", 1)

        result = tuner.tune(
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
        )

        best_params = result["best_params"]
        self.cfg.experiment.hyperparams.update(best_params)

        print("\n" + "=" * 60)
        print("  RETRAINING WITH BEST HYPERPARAMETERS")
        print("=" * 60 + "\n")

        training_pipeline = TrainingPipeline("")
        training_pipeline.cfg = self.cfg
        training_pipeline.mlflow_manager = self.mlflow_manager
        training_pipeline.run(data)

        print("\n" + "=" * 60)
        print("  TUNING PIPELINE COMPLETED")
        print("  Best model registered to MLflow Registry")
        print("=" * 60 + "\n")

        return best_params
