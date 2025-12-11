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
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.trainer.trainer_factory import TrainerFactory


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline supporting deep learning (DL) and
    traditional machine learning (ML) forecasting models.

    Responsibilities:
    - Preprocess dataset and save artifacts.
    - Initialize models and trainers.
    - Train models and evaluate on test sets.
    - Log metrics and models to MLflow when enabled.
    """

    def __init__(self, cfg_path: str = ""):
        """Initialize training pipeline and load configuration."""
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.mlflow_manager = MLflowManager(self.cfg)

    def preprocess(self) -> Any:
        """
        Fit preprocessing pipeline and transform dataset.

        Returns:
            Preprocessed dataset object.
        """
        preprocessor = OfflinePreprocessor(self.cfg, self.mlflow_manager)
        return preprocessor.run()

    def _init_model(self, approach: Dict[str, Any]):
        """
        Initialize model wrapper using ModelFactory.

        Args:
            approach: Dictionary containing model name and hyperparameters.

        Returns:
            Initialized model wrapper.
        """
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})
        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)
        return ModelFactory.create(name, hp)

    def _run_dl(
        self, trainer, dm: TSDLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """
        Train and evaluate a deep learning model.

        Returns:
            Tuple of (y_test, predictions)
        """
        train_loader, val_loader, _, _ = dm.get_loaders()
        wrapper = trainer.train(train_loader, val_loader, hyperparams)
        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)
        return y_test, preds

    def _run_ml(
        self, trainer, dm: TSMLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """
        Train and evaluate a traditional ML model.

        Returns:
            Tuple of (y_test, predictions)
        """
        x_train, y_train, x_val, y_val, x_test, y_test = dm.get_data()
        wrapper = trainer.train((x_train, y_train), (x_val, y_val), hyperparams)
        preds = wrapper.predict(x_test)
        return y_test, preds

    def _execute_training(self, trainer, dm, wrapper, hyperparams: Dict[str, Any]):
        """
        Execute training and evaluation, independent of MLflow.

        Returns:
            Evaluation metrics dictionary.
        """
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
        Retrieve a small sample input for model logging.

        Returns:
            Numpy array with first few test inputs.
        """
        if isinstance(dm, TSDLDataModule):
            x_test, _ = dm.get_test_windows()
            return x_test[:5]
        if isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, _ = dm.get_data()
            return x_test[:5]
        return None

    def run_approach(self, approach: Dict[str, Any], data: Any):
        """
        Execute a full training approach with optional MLflow logging.

        Steps:
        1. Preprocess dataset and save artifacts.
        2. Initialize model and trainer.
        3. Train and evaluate model.
        4. Log metrics and model to MLflow if enabled.

        Args:
            approach: Dictionary containing model name and hyperparameters.
            data: Raw dataset to train on.

        Returns:
            Metrics dictionary.
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

        if self.mlflow_manager:
            run_name = f"{model_name}_run"
            registry_name = (
                self.cfg.get("mlflow", {}).get("registry", {}).get("model_name")
            )

            with self.mlflow_manager.start_run(run_name=run_name):
                preprocessor_logger = OfflinePreprocessor(self.cfg, self.mlflow_manager)
                preprocessor_logger.log_artifacts_to_mlflow(df=df)

                metrics = self._execute_training(trainer, dm, wrapper, hyperparams)
                self.mlflow_manager.log_metrics(metrics)

                sample_input = self._get_sample_input(dm)

                self.mlflow_manager.log_model(
                    model_wrapper=wrapper,
                    artifact_path="model",
                    input_example=sample_input,
                    registered_model_name=registry_name,
                )

                if registry_name:
                    print(
                        f"[MLflow] Model registered as '{registry_name}' \
                          version 'latest'"
                    )

                return metrics

        return self._execute_training(trainer, dm, wrapper, hyperparams)
