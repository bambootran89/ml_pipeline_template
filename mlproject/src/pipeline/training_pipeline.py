from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.trainer_factory import TrainerFactory
from mlproject.src.utils.mlflow_manager import MLflowManager


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline for both deep learning and ML models.

    Responsibilities:
    - Preprocess data using OfflinePreprocessor
    - Build correct DataModule (DL or ML)
    - Initialize model via ModelFactory
    - Train model using TrainerFactory (DL or ML trainer)
    - Evaluate predictions with TimeSeriesEvaluator
    - Log experiments to MLflow (if enabled)
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        # Initialize MLflowManager if enabled in config
        self.mlflow = (
            MLflowManager(self.cfg)
            if self.cfg.get("mlflow", {}).get("enabled", False)
            else None
        )

    def preprocess(self):
        """Fit scaler once and return transformed dataset."""
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def _init_model(self, approach: Dict[str, Any]):
        """Initialize model wrapper from model factory."""
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.create(name, hp)

    def _run_dl(
        self, trainer, dm: TSDLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run deep learning training + evaluation."""
        train_loader, val_loader, _, _ = dm.get_loaders()

        wrapper = trainer.train(train_loader, val_loader, hyperparams)

        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)

        return y_test, preds

    def _run_ml(
        self, trainer, dm: TSMLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run ML model (sklearn/xgboost/lightgbm) training + evaluation."""
        x_train, y_train, x_val, y_val, x_test, y_test = dm.get_data()

        wrapper = trainer.train(
            (x_train, y_train),
            (x_val, y_val),
            hyperparams,
        )

        preds = wrapper.predict(x_test)
        return y_test, preds

    def _execute_training(self, trainer, dm, wrapper, hyperparams: Dict[str, Any]):
        """
        Execute the core training and evaluation logic.
        This method is separated to allow reuse within/without MLflow context.
        """
        # Dispatch based on DataModule class -> model agnostic logic
        if isinstance(dm, TSDLDataModule):
            y_test, preds = self._run_dl(trainer, dm, wrapper, hyperparams)

        elif isinstance(dm, TSMLDataModule):
            y_test, preds = self._run_ml(trainer, dm, wrapper, hyperparams)

        else:
            raise NotImplementedError(f"Unsupported DataModule type: {type(dm)}")

        metrics = TimeSeriesEvaluator().evaluate(y_test, preds)
        print(metrics)
        return metrics

    def _get_sample_input(self, dm):
        """
        Helper to retrieve a sample input (x_test) from DataModule
        for MLflow Model Signature inference.
        """
        if isinstance(dm, TSDLDataModule):
            x_test, _ = dm.get_test_windows()
            # Return a small batch or first 5 samples
            return x_test[:5]
        elif isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, _ = dm.get_data()
            return x_test[:5]
        return None

    def run_approach(self, approach: Dict[str, Any], data):
        """
        Train + evaluate one approach.
        Uses MLflow context manager to LOG METRICS & REGISTER MODEL.
        """

        df = data
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})

        wrapper = self._init_model(approach)
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        trainer = TrainerFactory.create(
            model_name, wrapper, self.cfg.training.artifacts_dir
        )

        # Optimization: Use MLflow Manager Context if enabled
        if self.mlflow:
            run_name = f"{model_name}_run"

            # Lấy tên model để register từ config (nếu có)
            # Ví dụ: config.yaml -> mlflow.registry.model_name: "ts_forecast_production"
            reg_name = self.cfg.get("mlflow", {}).get("registry", {}).get("model_name")

            with self.mlflow.start_run(run_name=run_name):
                # 1. Execute training & evaluation
                metrics = self._execute_training(trainer, dm, wrapper, hyperparams)

                # 2. Log Metrics
                print(f"[MLflow] Logging metrics for {run_name}...")
                self.mlflow.log_metrics(metrics)

                # 3. Log Model & Register to Registry
                sample_input = self._get_sample_input(dm)

                print(
                    f"[MLflow] Logging & Registering model artifact for {run_name}..."
                )
                self.mlflow.log_model(
                    model_wrapper=wrapper,
                    artifact_path="model",
                    input_example=sample_input,
                    registered_model_name=reg_name,  # <--- THÊM DÒNG NÀY
                )

                if reg_name:
                    print(f"[MLflow] Model registered as '{reg_name}' version 'latest'")

        else:
            # Run normally without MLflow
            self._execute_training(trainer, dm, wrapper, hyperparams)
