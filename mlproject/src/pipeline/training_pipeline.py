from typing import Any, Dict, Optional

import mlflow
from omegaconf import DictConfig

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification_eval import ClassificationEvaluator
from mlproject.src.eval.regression_eval import RegressionEvaluator
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tracking.pyfunc_preprocess import log_preprocessing_model
from mlproject.src.trainer.trainer_factory import TrainerFactory
from mlproject.src.utils.config_loader import ConfigLoader


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
        self.preprocessor = OfflinePreprocessor(is_train=True, cfg=self.cfg)
        self.evaluator: BaseEvaluator
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")
        if eval_type == "classification":
            self.evaluator = ClassificationEvaluator()
        elif eval_type == "regression":
            self.evaluator = RegressionEvaluator()
        else:
            self.evaluator = TimeSeriesEvaluator()

    def preprocess(self) -> Any:
        """
        Fit preprocessing pipeline and transform dataset.

        Returns:
            Preprocessed dataset object.
        """
        df = self.preprocessor.run()
        return df

    def _init_model(self, approach: Dict[str, Any]):
        """
        Initialize model wrapper using ModelFactory.

        Args:
            approach: Dictionary containing model name and hyperparameters.

        Returns:
            Initialized model wrapper.
        """
        name = approach["model"].lower()
        return ModelFactory.create(name, self.cfg)

    def _execute_training(self, trainer, dm, wrapper, hyperparams: Dict[str, Any]):
        """
        Execute training and evaluation, independent of MLflow.

        Returns:
            Evaluation metrics dictionary.
        """

        wrapper = trainer.train(dm, hyperparams)
        if hasattr(dm, "get_test_windows"):
            x_test, y_test = dm.get_test_windows()  # TSDLDataModule
        elif hasattr(dm, "get_data"):
            _, _, _, _, x_test, y_test = dm.get_data()  # TSMLDataModule
        else:
            raise AttributeError("DataModule must support retrieving test data")

        preds = wrapper.predict(x_test)
        metrics = self.evaluator.evaluate(y_test, preds)
        print(metrics)
        return metrics

    def _get_sample_input(self, dm):
        """
        Retrieve a small sample input for model logging.

        Returns:
            Numpy array with first few test inputs.
        """
        if hasattr(dm, "get_test_windows"):
            x_test, _ = dm.get_test_windows()
            return x_test[:5]
        else:
            _, _, _, _, x_test, _ = dm.get_data()
            return x_test[:5]

    def run_approach(self, approach: Dict[str, Any], data: Any) -> Dict[str, float]:
        """
        Execute a full training approach with optional MLflow logging.

        This method acts as a high-level orchestrator. It is responsible for:
        - Initializing model wrapper and data module
        - Creating the trainer
        - Delegating execution either to:
            * an MLflow-enabled training flow, or
            * a plain training flow without tracking

        Heavy logic (training, MLflow logging) is delegated to helper methods
        to keep this function simple and pylint-compliant.

        Args:
            approach (Dict[str, Any]):
                Dictionary describing the training approach.
                Expected keys include:
                - "model": model name
                - "hyperparams": optional hyperparameter dictionary
            data (Any):
                Raw dataset used to build the DataModule.

        Returns:
            Dict[str, float]:
                Dictionary of evaluation metrics produced by the training run.
        """
        df = data
        model_name: str = approach["model"].lower()
        hyperparams: Dict[str, Any] = approach.get("hyperparams", {})
        model_type: str = approach["model_type"].lower()
        wrapper = self._init_model(approach)
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        trainer = TrainerFactory.create(
            model_type=model_type,
            model_name=model_name,
            wrapper=wrapper,
            save_dir=self.cfg.training.artifacts_dir,
        )

        if self.mlflow_manager:
            return self._run_with_mlflow(
                model_name=model_name,
                trainer=trainer,
                dm=dm,
                wrapper=wrapper,
                hyperparams=hyperparams,
            )

        return self._execute_training(trainer, dm, wrapper, hyperparams)

    def _run_with_mlflow(
        self,
        *,
        model_name: str,
        trainer: Any,
        dm: Any,
        wrapper: Any,
        hyperparams: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Execute a training run with full MLflow tracking enabled.

        This method is responsible for:
        - Starting and managing an MLflow run
        - Logging preprocessing artifacts and preprocessing PyFunc model
        - Executing training and evaluation
        - Logging metrics and trained model to MLflow
        - Optionally registering the model in the MLflow Model Registry

        All MLflow-specific logic is intentionally isolated here to:
        - Reduce cognitive load in `run_approach`
        - Avoid pylint `too-many-locals`
        - Improve testability and maintainability

        Args:
            model_name (str):
                Normalized model name used for naming MLflow runs.
            trainer (Any):
                Initialized trainer responsible for fitting the model.
            dm (Any):
                Prepared DataModule instance.
            wrapper (Any):
                Model wrapper containing the trainable model.
            hyperparams (Dict[str, Any]):
                Hyperparameter dictionary passed to the training routine.

        Returns:
            Dict[str, float]:
                Dictionary of evaluation metrics logged to MLflow.
        """
        run_name: str = f"{model_name}_run"
        registry_name: Optional[str] = (
            self.cfg.get("mlflow", {}).get("registry", {}).get("model_name")
        )

        with self.mlflow_manager.start_run(run_name=run_name):
            transform_manager: Optional[
                TransformManager
            ] = self.preprocessor.transform_manager

            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError(
                    "[TrainingPipeline] No active MLflow run found while "
                    "logging preprocessing."
                )

            run_id: str = active_run.info.run_id

            if transform_manager is not None:
                log_preprocessing_model(
                    transform_manager=transform_manager,
                    run_id=run_id,
                    artifact_path="preprocessing_pipeline",
                )

            metrics: Dict[str, float] = self._execute_training(
                trainer, dm, wrapper, hyperparams
            )
            self.mlflow_manager.log_metrics(metrics)

            sample_input = self._get_sample_input(dm)

            self.mlflow_manager.log_model(
                model_wrapper=wrapper,
                artifact_path="model",
                input_example=sample_input,
                registered_model_name=registry_name,
            )

            return metrics
