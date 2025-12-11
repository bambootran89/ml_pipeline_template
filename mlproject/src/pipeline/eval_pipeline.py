"""
Evaluation Pipeline: Load model AND preprocessing artifacts từ MLflow.

Key changes:
1. Load scaler từ MLflow run artifacts
2. Fallback to local scaler nếu MLflow không có
3. Log evaluation với correct preprocessing version
"""

import os
import pickle
import tempfile

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.base import PreprocessBase
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager


class EvalPipeline(BasePipeline):
    """
    Evaluation pipeline that loads BOTH model AND preprocessing artifacts
    from MLflow Registry.

    Workflow:
    ---------
    1. Load configuration
    2. Load model from MLflow Registry
    3. ✅ Load preprocessing artifacts (scaler) from model's run
    4. Preprocess test data using SAME scaler as training
    5. Evaluate on test data
    6. Log evaluation metrics to MLflow
    """

    def __init__(self, cfg_path: str = ""):
        """Initialize evaluation pipeline."""
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)

        # Model registry name
        self.model_name = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )

        # ✅ Store loaded scaler for preprocessing
        self.scaler = None
        self.scaler_columns = None

    def _load_preprocessing_artifacts_from_mlflow(self, model_version_info) -> bool:
        """
        Load preprocessing scaler từ MLflow run artifacts.

        Args:
            model_version_info: ModelVersion object từ MLflow

        Returns:
            bool: True nếu load thành công, False nếu không
        """
        try:
            # Get run_id từ model version
            run_id = model_version_info.run_id

            print(f"[Evaluation] Loading preprocessing artifacts from run: {run_id}")

            # Download scaler artifact
            client = mlflow.MlflowClient()

            # Try to download scaler from preprocessing/scaler path
            try:
                artifact_path = "preprocessing/scaler/scaler.pkl"
                local_path = client.download_artifacts(run_id, artifact_path)

                # Load scaler
                with open(local_path, "rb") as f:
                    scaler_data = pickle.load(f)

                self.scaler = scaler_data.get("scaler")
                self.scaler_columns = scaler_data.get("columns")

                print(f"[Evaluation] ✅ Loaded scaler from MLflow run {run_id}")
                print(f"[Evaluation] Scaler features: {self.scaler_columns}")

                return True

            except Exception as e:
                print(f"[Evaluation] ⚠️  Could not load scaler from MLflow: {e}")
                return False

        except Exception as e:
            print(f"[Evaluation] ⚠️  Error loading preprocessing artifacts: {e}")
            return False

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess dataset using scaler từ MLflow (hoặc fallback to local).

        Returns:
            pd.DataFrame: Transformed dataset
        """
        # 1. Load raw data
        preprocessor = OfflinePreprocessor(self.cfg)
        df = preprocessor.load_raw_data()

        # 2. Apply basic preprocessing (fill missing, covariates)
        # Nhưng KHÔNG fit scaler mới
        engine = preprocessor.engine
        df = engine.base._apply_fill_missing(df)
        df = engine.base._apply_generate_covariates(df)

        # 3. ✅ Apply scaler từ MLflow (nếu đã load)
        if self.scaler is not None and self.scaler_columns is not None:
            print("[Evaluation] Using scaler from MLflow")

            # Ensure all scaler columns exist
            for col in self.scaler_columns:
                if col not in df.columns:
                    df[col] = 0.0

            # Transform using MLflow scaler
            df[self.scaler_columns] = self.scaler.transform(df[self.scaler_columns])
        else:
            # Fallback: load local scaler
            print("[Evaluation] ⚠️  Using LOCAL scaler (not from MLflow)")
            engine.base.load_scaler()
            df = engine.base._apply_scaling(df)

        return df

    def _load_model_from_registry(self, version: str = "latest"):
        """
        Load model từ MLflow Registry và download preprocessing artifacts.

        Args:
            version: Registry version or alias

        Returns:
            model: Loaded PyFunc model
            model_version_info: ModelVersion metadata
        """
        model_uri = f"models:/{self.model_name}/{version}"
        print(f"[Evaluation] Loading model from Registry URI: {model_uri}")

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # ✅ Get model version info để lấy run_id
        client = mlflow.MlflowClient()

        if version == "latest":
            # Get latest version number
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{self.model_name}'")

            # Sort by version number descending
            versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
            model_version_info = versions[0]
        else:
            model_version_info = client.get_model_version(self.model_name, version)

        print(f"[Evaluation] Model version: {model_version_info.version}")
        print(f"[Evaluation] Run ID: {model_version_info.run_id}")

        # ✅ Load preprocessing artifacts từ cùng run
        self._load_preprocessing_artifacts_from_mlflow(model_version_info)

        return model, model_version_info

    def run_approach(self, approach, data):
        """
        Execute evaluation loop:
        load model → load preprocessing → evaluate → log metrics.

        Args:
            approach: Unused (kept for compatibility)
            data: Raw or preprocessed dataset

        Returns:
            dict: Evaluation metrics
        """
        print(f"[{self.__class__.__name__}] Starting Registry-based Evaluation...")

        # 1. Load model từ registry (và load preprocessing artifacts)
        model, model_version_info = self._load_model_from_registry(version="latest")

        # 2. Preprocess data (using scaler từ MLflow)
        df = data

        # 3. Build DataModule
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        # 4. Get test dataset
        if isinstance(dm, TSDLDataModule):
            x_test, y_test = dm.get_test_windows()
        elif isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, y_test = dm.get_data()
        else:
            raise NotImplementedError(f"Unsupported DataModule: {type(dm)}")

        # 5. Ensure correct dtype
        x_test = np.asarray(x_test, dtype=np.float32)

        # 6. Run evaluation
        run_name = f"eval_{self.model_name}_v{model_version_info.version}"

        with self.mlflow_manager.start_run(run_name=run_name):
            print("[Evaluation] Predicting on test set...")

            preds = model.predict(x_test)

            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(y_test, preds)

            # Pretty-print metrics
            print("\n" + "=" * 60)
            print(f" EVALUATION METRICS - Model v{model_version_info.version}")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"{k:15} = {v:.6f}")
            print("=" * 60 + "\n")

            # Log metrics
            self.mlflow_manager.log_metrics(metrics)

            # ✅ Log which model version was evaluated
            self.mlflow_manager.log_params(
                {
                    "evaluated_model_name": self.model_name,
                    "evaluated_model_version": model_version_info.version,
                    "evaluated_run_id": model_version_info.run_id,
                    "preprocessing_from_mlflow": self.scaler is not None,
                }
            )

        return metrics
