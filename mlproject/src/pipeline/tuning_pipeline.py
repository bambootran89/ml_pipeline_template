#!/usr/bin/env python
"""
Tuning pipeline orchestrating hyperparameter search and final model retraining.

Workflow overview:
    1. Run Optuna-based hyperparameter tuning using time-series CV.
    2. Extract the best hyperparameters from the tuning study.
    3. Retrain the model using the full dataset with the best parameters.
    4. Register the retrained model into MLflow Model Registry.
"""

from typing import Any, Dict

from omegaconf import DictConfig

from mlproject.src.cv.splitter import ExpandingWindowSplitter
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.pipeline.training_pipeline import TrainingPipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna_tuner import OptunaTuner


class TuningPipeline(BasePipeline):
    """
    End-to-end tuning workflow combining:
        - Hyperparameter optimization
        - Best-parameter extraction
        - Final model retraining
        - Model registration in MLflow
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize tuning pipeline.

        Args:
            cfg_path:
                Path to the experiment configuration YAML file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)

        # Build CV splitter. Parameters may be overridden in YAML config.
        self.splitter = ExpandingWindowSplitter(
            n_splits=self.cfg.get("tuning", {}).get("n_splits", 3),
            test_size=self.cfg.get("tuning", {}).get("test_size", 20),
        )

    def preprocess(self):
        """
        Load and preprocess the dataset for tuning and retraining.

        Returns:
            Processed dataset produced by offline preprocessing workflow.
        """
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def run_approach(self, approach: Dict[str, Any], data):
        """
        Not used for tuning workflow.

        Raises:
            NotImplementedError:
                Always raised because tuning uses `run()`.
        """
        raise NotImplementedError("Use run() for tuning workflow.")

    def run(self, data=None):
        """
        Execute the full tuning workflow.

        Steps:
            1. Preprocess data if not provided.
            2. Run Optuna hyperparameter tuning with time-series CV.
            3. Update configuration with best-found hyperparameters.
            4. Retrain model on the full dataset using best parameters.
            5. Register retrained model in MLflow.

        Args:
            data:
                Optional preprocessed dataset. If None, preprocessing
                will be executed automatically.

        Returns:
            Dict[str, Any]: Best hyperparameters found by the tuner.
        """
        # Step 1: Preprocess
        if data is None:
            data = self.preprocess()

        # Step 2: Initialize the tuner
        tuner = OptunaTuner(
            cfg=self.cfg,
            splitter=self.splitter,
            mlflow_manager=self.mlflow_manager,
            metric_name=self.cfg.get("tuning", {}).get("optimize_metric", "mae_mean"),
            direction="minimize",
        )

        # Step 3: Run tuning
        n_trials = self.cfg.get("tuning", {}).get("n_trials", 20)
        result = tuner.tune(n_trials=n_trials)
        best_params = result["best_params"]

        # Step 4: Update experiment hyperparameters
        self.cfg.experiment.hyperparams.update(best_params)

        # Step 5: Retrain model using best parameters
        print(f"\n{'=' * 60}")
        print("  RETRAINING WITH BEST HYPERPARAMETERS")
        print(f"{'=' * 60}\n")

        training_pipeline = TrainingPipeline("")
        training_pipeline.cfg = self.cfg  # inject updated config

        # Retrain + auto-log to MLflow Registry
        training_pipeline.run(data)

        print(f"\n{'=' * 60}")
        print("  TUNING PIPELINE COMPLETED")
        print("  Best model registered to MLflow Registry")
        print(f"{'=' * 60}\n")

        return best_params
