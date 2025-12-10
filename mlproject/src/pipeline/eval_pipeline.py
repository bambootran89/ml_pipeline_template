"""
Evaluation Pipeline: Load the LATEST model from MLflow Model Registry.

This pipeline follows MLOps best practices:
- Registry-based Continuous Delivery (CD)
- Decoupled experiment configuration
- Reproducible evaluation runs logged to MLflow
- Strict dtype safety for MLflow PyFunc models
"""

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
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager


class EvalPipeline(BasePipeline):
    """
    Evaluation pipeline that fetches the latest trained model from the
    MLflow Model Registry and evaluates it on the saved test dataset.

    The pipeline supports any model wrapper that was logged using MLflow PyFunc.

    Workflow:
    ---------
    1. Load configuration (OmegaConf)
    2. Initialize MLflow manager
    3. Preprocess dataset using saved preprocessing artifacts
    4. Load model from MLflow Model Registry (latest or alias)
    5. Evaluate on test data
    6. Log evaluation metrics to MLflow

    Attributes:
        cfg (DictConfig): Loaded experiment configuration.
        mlflow_manager (MLflowManager): Wrapper for MLflow operations.
        model_name (str): The registry model name to load.
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize the evaluation pipeline.

        Args:
            cfg_path (str): Path to YAML configuration file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        # MLflow manager
        self.mlflow_manager = MLflowManager(self.cfg)

        # Determine registry model name
        self.model_name = (
            self.cfg.get("mlflow", {})
            .get("registry", {})
            .get("model_name", "ts_forecast_model")
        )

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess dataset using the stored scaler and transformations.

        Unlike training, this does NOT fit the scaler again.

        Returns:
            pd.DataFrame: Transformed dataset ready for evaluation.
        """
        preprocessor = OfflinePreprocessor(self.cfg)
        df = preprocessor.load_raw_data()
        df = preprocessor.transform(df)
        return df

    def _load_model_from_registry(self, version: str = "latest"):
        """
        Load the model wrapper from MLflow Model Registry.

        Args:
            version (str):
                Registry version or alias:
                - "latest"
                - "Production"
                - "Staging"
                - Specific version number (e.g. "5")

        Returns:
            mlflow.pyfunc.PyFuncModel: Loaded PyFunc model.
        """
        model_uri = f"models:/{self.model_name}/{version}"
        print(f"[Evaluation] Loading model from Registry URI: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def run_approach(self, approach, data):
        """
        Execute the evaluation loop:
        preprocess → create datamodule → load model → evaluate → log metrics.

        Args:
            approach (Any): Unused. Kept for backward compatibility.
            data (pd.DataFrame): Raw or preprocessed dataset.

        Returns:
            dict: Evaluation metrics logged to MLflow.
        """
        print(f"[{self.__class__.__name__}] Starting Registry-based Evaluation...")

        # 1. Dataset construction
        df = data
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        # 2. Load model from registry
        model = self._load_model_from_registry(version="latest")

        # 3. Get test dataset
        if isinstance(dm, TSDLDataModule):
            x_test, y_test = dm.get_test_windows()
        elif isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, y_test = dm.get_data()
        else:
            raise NotImplementedError(f"Unsupported DataModule: {type(dm)}")

        # 4. PyFunc-compatible dtype
        x_test = np.asarray(x_test, dtype=np.float32)

        # 5. MLflow evaluation run
        run_name = f"eval_{self.model_name}_latest"
        with self.mlflow_manager.start_run(run_name=run_name):
            print("[Evaluation] Predicting on test set...")

            preds = model.predict(x_test)

            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(y_test, preds)

            # Pretty-print metrics
            print("\n" + "=" * 40)
            print(" FINAL TEST METRICS (Registry Model) ")
            print("=" * 40)
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("=" * 40 + "\n")

            self.mlflow_manager.log_metrics(metrics)

        return metrics
