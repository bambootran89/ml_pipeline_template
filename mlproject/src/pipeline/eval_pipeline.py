"""
Evaluation pipeline: Load the latest model from MLflow Model Registry
and evaluate it on the test dataset.
"""

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification_eval import ClassificationEvaluator
from mlproject.src.eval.regression_eval import RegressionEvaluator
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.utils.func_utils import (
    flatten_metrics_for_mlflow,
    load_model_from_registry,
    sync_artifacts_from_registry,
)


class EvalPipeline(BasePipeline):
    """
    Evaluation pipeline for models stored in MLflow Model Registry.
    Downloads preprocessing artifacts and evaluates model performance
    on the test dataset.
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize the pipeline. Sync artifacts and load the model if
        MLflow is enabled.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.model = None
        self.mlflow_manager = MLflowManager(self.cfg)
        self.model_name = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )
        if self.mlflow_manager.enabled:
            sync_artifacts_from_registry(
                self.model_name, self.cfg.preprocessing.artifacts_dir
            )
            self.model = load_model_from_registry(self.model_name, version="latest")
        else:
            print("MLflow disabled. Cannot load model from registry.")

        self.preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
        self.evaluator: BaseEvaluator
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")
        if eval_type == "classification":
            self.evaluator = ClassificationEvaluator()
        elif eval_type == "regression":
            self.evaluator = RegressionEvaluator()
        else:
            self.evaluator = TimeSeriesEvaluator()

    def preprocess(self) -> pd.DataFrame:
        """
        Load raw data and apply preprocessing transformations.
        Returns a preprocessed DataFrame.
        """
        df = self.preprocessor.load_raw_data()
        fea_df = self.preprocessor.engine.offline_transform(df)
        data_cfg = self.cfg.get("data", {})
        data_type = data_cfg.get("type", "timeseries").lower()

        if data_type == "timeseries":
            return fea_df
        else:
            target_cols = data_cfg.get("target_columns", [])
            tar_df = df[target_cols]
            return pd.concat([fea_df, tar_df], axis=1)

    def run_approach(self, approach: Any, data: pd.DataFrame):
        """
        Evaluate the model on the test dataset.
        Returns a dictionary of metrics.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Check MLflow connection or pipeline initialization."
            )

        df = data
        dm = DataModuleFactory.build(self.cfg, df)
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
        print(safe_metrics)
        return metrics
