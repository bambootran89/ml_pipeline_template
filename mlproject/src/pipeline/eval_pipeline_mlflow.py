"""
Evaluation Pipeline với MLflow - load model từ Model Registry.
"""
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


class EvalPipelineMLflow(BasePipeline):
    """
    Evaluation pipeline load model từ MLflow Model Registry.
    """

    def __init__(
        self,
        cfg_path: str = "",
        model_name: str = "",
        model_version: str = "latest",
    ):
        """
        Args:
            cfg_path: Path to config
            model_name: Tên model trong registry (nếu None, dùng từ config)
            model_version: Version của model ("latest", "1", "2", ...)
        """
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)

        # Model info
        if not model_name:
            model_name = self.cfg.mlflow.get("registry", {}).get(
                "model_name", "ts_forecast_model"
            )
        self.model_name = model_name
        self.model_version = model_version

    def preprocess(self):
        """Transform data bằng saved scaler."""
        preprocessor = OfflinePreprocessor(self.cfg)
        df = preprocessor.load_raw_data()
        df = preprocessor.transform(df)
        return df

    def _load_model_from_registry(self):
        """
        Load model từ MLflow Model Registry.

        Returns:
            Loaded model wrapper
        """
        # Construct model URI
        if self.model_version == "latest":
            model_uri = f"models:/{self.model_name}/latest"
        else:
            model_uri = f"models:/{self.model_name}/{self.model_version}"

        print(f"Loading model from: {model_uri}")

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        return model

    def run_approach(self, approach, data):
        """
        Evaluate model từ MLflow registry.
        """
        # Start evaluation run
        self.mlflow_manager.start_run(
            run_name=f"eval_{self.model_name}_v{self.model_version}"
        )

        try:
            df = data
            dm = DataModuleFactory.build(self.cfg, df)
            dm.setup()

            # Load model từ registry
            model = self._load_model_from_registry()

            # Get test data
            if isinstance(dm, TSDLDataModule):
                x_test, y_test = dm.get_test_windows()
            elif isinstance(dm, TSMLDataModule):
                _, _, _, _, x_test, y_test = dm.get_data()
            else:
                raise NotImplementedError(f"Unsupported DataModule: {type(dm)}")

            # CRITICAL FIX: Ensure float32 dtype
            x_test = np.asarray(x_test, dtype=np.float32)

            # Predict
            preds = model.predict(x_test)

            # Evaluate
            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(y_test, preds)

            print("\n=== Evaluation Metrics ===")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            # Log metrics
            self.mlflow_manager.log_metrics(metrics)

        finally:
            self.mlflow_manager.end_run()
