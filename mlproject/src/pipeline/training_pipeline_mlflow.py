"""
Training Pipeline tích hợp MLflow tracking và model registry.
"""
import os
from typing import Any, Dict

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


class TrainingPipelineMLflow(BasePipeline):
    """
    Training pipeline với MLflow integration.

    Chức năng:
    - Track experiments với MLflow
    - Log parameters, metrics, artifacts
    - Register model vào Model Registry
    - Auto-versioning models
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(self.cfg)

    def preprocess(self):
        """Fit scaler và transform dataset."""
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def _init_model(self, approach: Dict[str, Any]):
        """Initialize model wrapper."""
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.create(name, hp)

    def _run_dl(
        self, trainer, dm: TSDLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run DL training với MLflow logging."""
        train_loader, val_loader, _, _ = dm.get_loaders()

        # Train model
        wrapper = trainer.train(train_loader, val_loader, hyperparams)

        # Evaluate
        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)

        # Log model với input example
        self.mlflow_manager.log_model(
            wrapper,
            artifact_path="model",
            input_example=x_test[:1],  # Log 1 example
        )

        return y_test, preds

    def _run_ml(
        self, trainer, dm: TSMLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run ML training với MLflow logging."""
        x_train, y_train, x_val, y_val, x_test, y_test = dm.get_data()

        # Train model
        wrapper = trainer.train(
            (x_train, y_train),
            (x_val, y_val),
            hyperparams,
        )

        # Evaluate
        preds = wrapper.predict(x_test)

        # Log model với input example
        x_example = x_test[:1].reshape(1, -1) if x_test.ndim > 2 else x_test[:1]
        self.mlflow_manager.log_model(
            wrapper,
            artifact_path="model",
            input_example=x_example,
        )

        return y_test, preds

    def run_approach(self, approach: Dict[str, Any], data):
        """
        Train + evaluate với MLflow tracking.
        """
        # Start MLflow run
        run_name = f"{approach['name']}_{approach['model']}"
        self.mlflow_manager.start_run(run_name=run_name)

        try:
            df = data
            model_name = approach["model"].lower()
            hyperparams = approach.get("hyperparams", {})

            # Log hyperparameters
            self.mlflow_manager.log_params(
                {
                    "model": model_name,
                    "experiment_type": approach.get("type", "unknown"),
                    **hyperparams,
                }
            )

            # Initialize components
            wrapper = self._init_model(approach)
            dm = DataModuleFactory.build(self.cfg, df)
            dm.setup()

            # Log data split info
            n_train, n_val, n_test = dm.summary()
            self.mlflow_manager.log_params(
                {
                    "n_train": n_train,
                    "n_val": n_val,
                    "n_test": n_test,
                }
            )

            trainer = TrainerFactory.create(
                model_name, wrapper, self.cfg.training.artifacts_dir
            )

            # Train và evaluate
            if isinstance(dm, TSDLDataModule):
                y_test, preds = self._run_dl(trainer, dm, wrapper, hyperparams)
            elif isinstance(dm, TSMLDataModule):
                y_test, preds = self._run_ml(trainer, dm, wrapper, hyperparams)
            else:
                raise NotImplementedError(f"Unsupported DataModule: {type(dm)}")

            # Evaluate và log metrics
            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(y_test, preds)

            print("\n=== Evaluation Metrics ===")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            self.mlflow_manager.log_metrics(metrics)

            # Log scaler artifact
            scaler_path = os.path.join(
                self.cfg.preprocessing.artifacts_dir, "scaler.pkl"
            )
            self.mlflow_manager.log_scaler(scaler_path)

            # Register model vào Model Registry
            model_uri = f"runs:/{self.mlflow_manager.run_id}/model"
            model_version = self.mlflow_manager.register_model(
                model_uri=model_uri,
                model_name=f"{model_name}_forecaster",
            )

            if model_version:
                print(
                    f"\n✓ Model registered: {model_version.name} v{model_version.version}"
                )

        except Exception as e:
            print(f"Error during training: {e}")
            raise

        finally:
            # Always end run
            self.mlflow_manager.end_run()
