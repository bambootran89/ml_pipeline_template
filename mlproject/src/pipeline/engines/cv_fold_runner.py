from typing import Any, Tuple

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.trainer.trainer_factory import TrainerFactory

from ...eval.fold_evaluator import FoldEvaluator


class FoldRunner:
    """
    Execute a complete workflow for a single cross-validation fold.

    This class handles preprocessing, datamodule creation, model construction,
    training, evaluation, and MLflow logging. Heavy lifting is delegated to
    modular helper components to keep the fold pipeline maintainable.
    """

    def __init__(self, cfg: Any, mlflow_manager: MLflowManager):
        self.cfg = cfg
        self.mlflow_log_all = False
        self.evaluator = FoldEvaluator()
        self.mlflow_manager = mlflow_manager

    def _extract_test_data(self, dm) -> Tuple[Any, Any]:
        """
        Extract test windows from the datamodule,
          supporting both TSDLDataModule and TSMLDataModule.

        Parameters
        ----------
        dm : Any
            Datamodule instance created by DataModuleFactory.

        Returns
        -------
        Tuple[Any, Any]
            Test features (x_test) and test targets (y_test).
        """
        if hasattr(dm, "get_test_windows"):
            return dm.get_test_windows()

        if hasattr(dm, "get_data"):
            # TSMLDataModule returns 6 outputs
            _, _, _, _, x_test, y_test = dm.get_data()
            return x_test, y_test

        raise RuntimeError("Datamodule does not provide test data extraction methods.")

    def _preprocess_fold(self, df_fold: Any) -> Any:
        """
        Preprocess raw fold dataframe.

        Parameters
        ----------
        df_fold : Any
            Raw input dataframe for the fold.

        Returns
        -------
        Any
            Preprocessed dataframe.
        """
        preprocessor = OfflinePreprocessor(self.cfg, self.mlflow_manager)
        df_processed = preprocessor.run(df_fold)
        return df_processed

    def _build_components(
        self,
        model_name: str,
        hyperparams: dict,
        df_processed: Any,
        is_tuning: bool,
    ):
        """
        Build model wrapper, trainer, and datamodule.

        Parameters
        ----------
        model_name : str
            Model identifier.
        hyperparams : dict
            Model hyperparameters.
        df_processed : Any
            Preprocessed dataframe.
        is_tuning : bool
            Disables artifacts during tuning.

        Returns
        -------
        tuple
            (model_wrapper, trainer, datamodule, preprocessor)
        """
        model_wrapper = ModelFactory.create(model_name, hyperparams)

        trainer = TrainerFactory.create(
            model_name=model_name,
            wrapper=model_wrapper,
            save_dir=self.cfg.training.artifacts_dir,
        )

        if is_tuning and hasattr(trainer, "artifacts_dir"):
            trainer.artifacts_dir = None

        datamodule = DataModuleFactory.build(self.cfg, df_processed)
        datamodule.setup()

        return trainer, datamodule

    def _train_model(
        self,
        trainer: Any,
        datamodule: Any,
        hyperparams: dict,
    ):
        """
        Train model on the given datamodule.

        Parameters
        ----------
        trainer : Any
            Trainer instance.
        datamodule : Any
            Prepared datamodule.
        hyperparams : dict
            Model hyperparameters.

        Returns
        -------
        Any
            Trained model instance.
        """
        model_trained = trainer.train(datamodule, hyperparams)
        return model_trained

    def _evaluate_model(
        self,
        model_trained: Any,
        datamodule: Any,
    ):
        """
        Run model evaluation.

        Parameters
        ----------
        model_trained : Any
            Trained model.
        datamodule : Any
            Test datamodule providing test split.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        x_test, y_test = self._extract_test_data(datamodule)
        metrics = self.evaluator.evaluate(model_trained, (x_test, y_test))
        return metrics, x_test

    def _log_fold_results(
        self,
        model_name: str,
        model_trained: Any,
        x_sample: Any,
        metrics: dict,
        preprocessor: Any,
    ):
        """
        Log metrics, artifacts, and model to MLflow.

        Parameters
        ----------
        model_name : str
            Name of model.
        model_trained : Any
            Trained model.
        x_sample : Any
            Example input batch.
        metrics : dict
            Evaluation results.
        preprocessor : Any
            Preprocessor with artifacts.
        """
        if not self.mlflow_manager:
            return

        run_name = f"{model_name}_run"
        with self.mlflow_manager.start_run(run_name=run_name, nested=True):
            self.mlflow_manager.log_metrics(metrics)
            preprocessor.log_artifacts_to_mlflow()

            self.mlflow_manager.log_model(
                model_wrapper=model_trained,
                artifact_path="model",
                input_example=x_sample,
            )

    def run_fold(
        self,
        df_fold: Any,
        model_name: str,
        hyperparams: dict,
        is_tuning: bool = False,
    ) -> dict:
        """
        Run a complete CV fold:
        preprocess → build → train → evaluate → log.

        Parameters
        ----------
        df_fold : Any
            Raw dataframe for this fold.
        model_name : str
            Model identifier.
        hyperparams : dict
            Hyperparameters.
        is_tuning : bool
            Disable artifact saving for tuning sweeps.

        Returns
        -------
        dict
            Fold evaluation metrics.
        """
        df_processed = self._preprocess_fold(df_fold)

        trainer, datamodule = self._build_components(
            model_name=model_name,
            hyperparams=hyperparams,
            df_processed=df_processed,
            is_tuning=is_tuning,
        )

        model_trained = self._train_model(trainer, datamodule, hyperparams)

        metrics, x_test = self._evaluate_model(model_trained, datamodule)

        # small test batch for logging
        x_sample = x_test[:5]

        # Logging (optional)
        preprocessor = OfflinePreprocessor(self.cfg, self.mlflow_manager)
        self._log_fold_results(
            model_name=model_name,
            model_trained=model_trained,
            x_sample=x_sample,
            metrics=metrics,
            preprocessor=preprocessor,
        )

        return metrics
