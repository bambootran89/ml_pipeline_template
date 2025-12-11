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
    End-to-end training pipeline supporting both deep learning (DL) and
    traditional machine learning (ML) forecasting models.

    Responsibilities:
        • Preprocess raw data using `OfflinePreprocessor`
        • Build the correct DataModule dynamically (DL or ML)
        • Initialize model wrapper through `ModelFactory`
        • Train using `TrainerFactory` (DL trainers or ML trainers)
        • Evaluate predictions via `TimeSeriesEvaluator`
        • Log metrics and register models to MLflow Model Registry

    This pipeline is designed for modularity, reproducibility,
    and MLOps-aligned experiment tracking.
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize training pipeline and load configuration.

        Args:
            cfg_path (str): Path to YAML config file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)

    def preprocess(self):
        """
        Fit preprocessing pipeline and transform the dataset.

        NOTE: Preprocessing artifacts được log trong run_approach(),
        không log ở đây để tránh nested run conflict.

        Returns:
            pd.DataFrame: Fully transformed dataset ready for DataModule.
        """
        # ✅ Pass mlflow_manager nhưng artifacts sẽ được log trong active run
        preprocessor = OfflinePreprocessor(self.cfg, self.mlflow_manager)
        return preprocessor.run()

    def _init_model(self, approach: Dict[str, Any]):
        """
        Initialize model wrapper using ModelFactory.

        Args:
            approach (dict): Model configuration containing:
                - model: model name key
                - hyperparams: hyperparameter dictionary

        Returns:
            ModelWrapper: The initialized model.
        """
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.create(name, hp)

    def _run_dl(
        self,
        trainer,
        dm: TSDLDataModule,
        wrapper,
        hyperparams: Dict[str, Any],
    ):
        """
        Train and evaluate a deep learning model.

        Args:
            trainer: DL trainer instance.
            dm (TSDLDataModule): Data module providing PyTorch dataloaders.
            wrapper: Model wrapper instance.
            hyperparams (dict): Training hyperparameters.

        Returns:
            tuple(np.ndarray, np.ndarray):
                - y_test: Ground truth test labels.
                - preds: Model predictions.
        """
        train_loader, val_loader, _, _ = dm.get_loaders()

        wrapper = trainer.train(train_loader, val_loader, hyperparams)

        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)

        return y_test, preds

    def _run_ml(
        self,
        trainer,
        dm: TSMLDataModule,
        wrapper,
        hyperparams: Dict[str, Any],
    ):
        """
        Train and evaluate a traditional ML model (XGBoost / LightGBM / sklearn).

        Args:
            trainer: ML trainer instance.
            dm (TSMLDataModule): ML dataset container.
            wrapper: ML model wrapper.
            hyperparams (dict): Model hyperparameters.

        Returns:
            tuple(np.ndarray, np.ndarray):
                - y_test: Ground truth values.
                - preds: Predictions from ML model.
        """
        x_train, y_train, x_val, y_val, x_test, y_test = dm.get_data()

        wrapper = trainer.train(
            (x_train, y_train),
            (x_val, y_val),
            hyperparams,
        )

        preds = wrapper.predict(x_test)
        return y_test, preds

    def _execute_training(
        self,
        trainer,
        dm,
        wrapper,
        hyperparams: Dict[str, Any],
    ):
        """
        Execute training + evaluation, independent of MLflow context.

        Args:
            trainer: Trainer class (DL or ML).
            dm: Corresponding DataModule (TSDL or TSML).
            wrapper: Model wrapper.
            hyperparams (dict): Hyperparameters for training.

        Returns:
            dict: Computed evaluation metrics.
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
        Retrieve a small input example from DataModule
        to construct MLflow model signatures.

        Args:
            dm: DataModule instance.

        Returns:
            np.ndarray | None: A small sample batch.
        """
        if isinstance(dm, TSDLDataModule):
            x_test, _ = dm.get_test_windows()
            return x_test[:5]

        if isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, _ = dm.get_data()
            return x_test[:5]

        return None

    def run_approach(self, approach: Dict[str, Any], data):
        """
        Execute one full training approach (model + hyperparameters).

        MLflow workflow:
            1. Start experiment run
            2. Preprocess + log artifacts (trong cùng run)
            3. Train model
            4. Log metrics
            5. Log model artifacts
            6. Register model (optional)

        Args:
            approach (dict):
                Model configuration block containing:
                    - model
                    - hyperparams
            data (pd.DataFrame): Preprocessed dataset.

        Returns:
            dict: Evaluation metrics.
        """
        df = data
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})

        wrapper = self._init_model(approach)
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        trainer = TrainerFactory.create(
            model_name,
            wrapper,
            self.cfg.training.artifacts_dir,
        )

        if self.mlflow_manager:
            run_name = f"{model_name}_run"

            registry_name = (
                self.cfg.get("mlflow", {}).get("registry", {}).get("model_name")
            )

            # ✅ START RUN - preprocessing artifacts sẽ được log trong run này
            with self.mlflow_manager.start_run(run_name=run_name):
                # Preprocessing artifacts đã được log trong preprocess()
                # (do OfflinePreprocessor có mlflow_manager và check active_run)

                metrics = self._execute_training(
                    trainer,
                    dm,
                    wrapper,
                    hyperparams,
                )

                print(f"[MLflow] Logging metrics for {run_name}...")
                self.mlflow_manager.log_metrics(metrics)

                sample_input = self._get_sample_input(dm)

                print(f"[MLflow] Logging & Registering model for {run_name}...")
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

        # Fallback: Run without MLflow
        return self._execute_training(
            trainer,
            dm,
            wrapper,
            hyperparams,
        )
