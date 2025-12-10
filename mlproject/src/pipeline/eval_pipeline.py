"""
Evaluation pipeline: Loads the LATEST model from MLflow Model Registry.
Refactored to align with MLOps best practices (Registry-based CD).
"""
from typing import Optional

import mlflow
import numpy as np
from omegaconf import DictConfig

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.utils.mlflow_manager import MLflowManager


class EvalPipeline(BasePipeline):
    """
    Evaluation Pipeline that fetches the production/latest model
    directly from MLflow Model Registry.
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        # Initialize MLflow Manager
        self.mlflow_manager = MLflowManager(self.cfg)

        # Determine Model Name for Registry
        # Priority: Config -> Default
        self.model_name = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )

    def preprocess(self):
        """
        Transform data using SAVED scaler (does NOT fit).

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        preprocessor = OfflinePreprocessor(self.cfg)

        df = preprocessor.load_raw_data()
        df = preprocessor.transform(df)
        return df

    def _load_model_from_registry(self, version: str = "latest"):
        """
        Load model wrapper from MLflow Model Registry.

        Args:
            version: Model version alias ("latest", "Production", "Staging")
                     or version number string.
        """
        model_uri = f"models:/{self.model_name}/{version}"
        print(f"[Evaluation] Loading model from Registry URI: {model_uri}")

        # mlflow.pyfunc.load_model returns the PyFunc wrapper
        # which wraps our custom ModelWrapper (via MLflowModelWrapper)
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    def run_approach(self, approach, data):
        """
        Execute evaluation.

        Args:
            run_id: Ignored in this version (kept for compatibility).
                    We always fetch 'latest' from registry.
        """
        print(f"[{self.__class__.__name__}] Starting Registry-based Evaluation...")

        df = data
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        # 3. Load Model (Latest from Registry)
        model = self._load_model_from_registry(version="latest")

        # 4. Get Test Data
        if isinstance(dm, TSDLDataModule):
            x_test, y_test = dm.get_test_windows()
        elif isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, y_test = dm.get_data()
        else:
            raise NotImplementedError(f"Unsupported DataModule: {type(dm)}")

        # 5. CRITICAL FIX: Ensure float32 dtype for PyFunc compatibility
        # MLflow/PyFunc models are strict about input types (often expecting float32)
        x_test = np.asarray(x_test, dtype=np.float32)

        # 6. Evaluation Run Tracking
        run_name = f"eval_{self.model_name}_latest"

        # Use Context Manager (Updated MLflowManager style)
        with self.mlflow_manager.start_run(run_name=run_name):
            print("[Evaluation] Predicting on test set...")

            # Predict
            preds = model.predict(x_test)

            # Calculate Metrics
            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(y_test, preds)

            print("\n" + "=" * 30)
            print(" FINAL TEST METRICS (Registry Model) ")
            print("=" * 30)
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("=" * 30 + "\n")

            # Log metrics to this new Eval Run
            self.mlflow_manager.log_metrics(metrics)
