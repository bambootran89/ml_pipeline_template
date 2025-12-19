"""
Evaluation pipeline.

Responsibilities:
- Load prediction model from MLflow Model Registry
- Load companion preprocessing model (if available)
- Preprocess test dataset
- Run evaluation and log metrics to MLflow
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.dataset_resolver import resolve_datasets_from_cfg
from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification_eval import ClassificationEvaluator
from mlproject.src.eval.clustering_eval import ClusteringEvaluator
from mlproject.src.eval.regression_eval import RegressionEvaluator
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow
from mlproject.src.utils.mlflow_utils import (
    load_companion_preprocessor_from_model,
    load_model_from_registry_safe,
)


class EvalPipeline(BasePipeline):
    """
    Evaluation pipeline for models stored in MLflow Model Registry.
    """

    def __init__(self, cfg_path: str = "") -> None:
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)
        self.model_name: str = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )

        self.model: Any = None
        self.preprocessor_model: Optional[Any] = None

        self.preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
        self.evaluator: BaseEvaluator = self._build_evaluator()

        if self.mlflow_manager.enabled:
            self.model = self._load_model_from_mlflow()

    def _load_model_from_mlflow(self) -> Any:
        """
        Load the prediction model from MLflow Model Registry and resolve its
        companion preprocessing model if available.

        This method performs the following steps:
        1. Loads the latest version of the prediction model from the MLflow
        Model Registry using the configured model name.
        2. Attempts to load the associated preprocessing PyFunc model using
        the run_id stored in the prediction model metadata.
        3. Falls back to local preprocessing logic if the companion
        preprocessing model cannot be resolved.

        Returns
        -------
        Any
            Loaded MLflow prediction model.

        Raises
        ------
        RuntimeError
            If the prediction model cannot be loaded from the MLflow
            Model Registry.
        """
        model = load_model_from_registry_safe(
            cfg=self.cfg,
            default_model_name=self.model_name,
        )

        if model is None:
            raise RuntimeError("Failed to load model from MLflow Registry")

        self.preprocessor_model = load_companion_preprocessor_from_model(model)

        if self.preprocessor_model is None:
            print(
                "[EvalPipeline] WARNING: Companion preprocessor not found. "
                "Using local preprocessing."
            )

        return model

    def preprocess(self) -> pd.DataFrame:
        """
        Load raw data and apply preprocessing.

        Priority:
        1. MLflow PyFunc preprocessing model
        2. Local OfflinePreprocessor fallback

        Returns
        -------
        pd.DataFrame
            Preprocessed dataset.
        """
        df, _, _, df_raw = resolve_datasets_from_cfg(self.cfg)
        is_use_dataset = True
        if len(df) > 0:
            df_raw = df.copy()
            is_use_dataset = False

        fea_df: pd.DataFrame = self._transform_data(df_raw)
        df = self._attach_targets_if_needed(df_raw, fea_df)
        if is_use_dataset:
            df["dataset"] = "test"

        return df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """

        if self.preprocessor_model is not None:
            print("[EvalPipeline] Using MLflow preprocessing model")
            return self.preprocessor_model.predict(df)
        else:
            print("[EvalPipeline] Using local preprocessing fallback")
            self.preprocessor.transform_manager.load(self.cfg)
            return self.preprocessor.transform(df)

    def _attach_targets_if_needed(
        self, df_raw: pd.DataFrame, fea_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach target columns back for tabular datasets.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data.
        fea_df : pd.DataFrame
            Transformed features.

        Returns
        -------
        pd.DataFrame
            Final dataset for evaluation.
        """
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", "timeseries")).lower()

        if data_type == "timeseries":
            return fea_df

        target_cols = data_cfg.get("target_columns", [])
        tar_df = df_raw[target_cols]

        return pd.concat([fea_df, tar_df], axis=1)

    # Evaluation

    def _build_evaluator(self) -> BaseEvaluator:
        """
        Build evaluator based on configuration.

        Returns
        -------
        BaseEvaluator
            Evaluator instance.
        """
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")

        if eval_type == "classification":
            return ClassificationEvaluator()
        elif eval_type == "regression":
            return RegressionEvaluator()

        elif eval_type == "clustering":
            return ClusteringEvaluator()
        elif eval_type == "timeseries":
            return TimeSeriesEvaluator()
        else:
            raise ValueError(f"Don't support this type {eval_type}")

    def run_approach(self, approach: Any, data: pd.DataFrame) -> dict:
        """
        Evaluate model on test dataset.

        Parameters
        ----------
        approach : Any
            Experiment configuration.
        data : pd.DataFrame
            Preprocessed dataset.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        dm = DataModuleFactory.build(self.cfg, data)
        dm.setup()

        if hasattr(dm, "get_test_windows"):
            x_test, y_test = dm.get_test_windows()
        else:
            _, _, _, _, x_test, y_test = dm.get_data()

        x_test = np.asarray(x_test, dtype=np.float32)

        run_name = f"eval_{self.model_name}_latest"

        with self.mlflow_manager.start_run(run_name=run_name):
            preds = self.model.predict(x_test)
            metrics = self.evaluator.evaluate(y_test, preds)
            safe_metrics = flatten_metrics_for_mlflow(metrics)
            self.mlflow_manager.log_metrics(safe_metrics)
        return metrics
