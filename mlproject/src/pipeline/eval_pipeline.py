"""
Evaluation pipeline: Load the latest model from MLflow Model Registry
and evaluate it on the test dataset.
"""

import os
import shutil
import tempfile
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification_eval import ClassificationEvaluator
from mlproject.src.eval.regression_eval import RegressionEvaluator
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow


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

        self.mlflow_manager = MLflowManager(self.cfg)
        self.model_name = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )
        self.model = None
        self.preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
        self.evaluator: BaseEvaluator
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")
        if eval_type == "classification":
            self.evaluator = ClassificationEvaluator()
        elif eval_type == "regression":
            self.evaluator = RegressionEvaluator()
        else:
            self.evaluator = TimeSeriesEvaluator()

        if self.mlflow_manager.enabled:
            self._sync_artifacts_from_registry()
            self.model = self._load_model_from_registry(version="latest")
        else:
            print("MLflow disabled. Cannot load model from registry.")

    def _sync_artifacts_from_registry(self):
        """
        Download preprocessing artifacts (scaler) from the latest run
        associated with the registered model.
        """
        client = MlflowClient()

        try:
            versions = client.get_latest_versions(self.model_name, stages=None)
            if not versions:
                print(f"No registered model found for '{self.model_name}'")
                return

            latest_version = max(versions, key=lambda v: int(v.version))
            run_id = latest_version.run_id

            local_artifacts_dir = self.cfg.preprocessing.artifacts_dir
            os.makedirs(local_artifacts_dir, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                client.download_artifacts(run_id, "preprocessing", tmp_dir)

                scaler_src = None
                for root, _, files in os.walk(tmp_dir):
                    if "scaler.pkl" in files:
                        scaler_src = os.path.join(root, "scaler.pkl")
                        break

                if scaler_src:
                    dst = os.path.join(local_artifacts_dir, "scaler.pkl")
                    shutil.copy2(scaler_src, dst)
                else:
                    print("scaler.pkl not found in Run artifacts.")

        except Exception as e:
            print(f"Error downloading artifacts: {e}")

    def _load_model_from_registry(self, version: str = "latest"):
        """
        Load the MLflow PyFunc model from the model registry.
        Returns the loaded model object.
        """
        model_uri = f"models:/{self.model_name}/{version}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

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
