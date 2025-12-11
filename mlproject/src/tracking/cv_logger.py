import tempfile

from mlproject.src.preprocess.base import PreprocessBase
from mlproject.src.tracking.mlflow_manager import MLflowManager


class FoldLogger:
    """
    Handles MLflow logging for a single cross-validation fold.

    This class isolates experiment tracking logic from FoldRunner and
    centralizes the logging of parameters, metrics, and preprocessing
    artifacts (e.g., fitted scalers).
    """

    def __init__(self, mlflow_manager: MLflowManager | None):
        self.manager = mlflow_manager

    def log(
        self,
        fold_num,
        model_name,
        hyperparams,
        train_idx,
        test_idx,
        metrics,
        preprocessor: PreprocessBase,
    ):
        """
        Log fold metadata, evaluation metrics, and preprocessing artifacts.

        Parameters
        ----------
        fold_num : int
            Index of the current cross-validation fold.
        model_name : str
            Name of the model being evaluated.
        hyperparams : dict
            Hyperparameters used in the fold.
        train_idx : Any
            Training sample indices.
        test_idx : Any
            Test/validation sample indices.
        metrics : dict
            Evaluation metrics from this fold.
        preprocessor : PreprocessBase
            Preprocessor used for fitting and transforming the fold data.
        """
        if not (self.manager and self.manager.enabled):
            return

        with self.manager.start_run(
            run_name=f"{model_name}_fold_{fold_num}", nested=True
        ):
            self.manager.log_params(
                {
                    "fold": fold_num,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **hyperparams,
                }
            )
            self.manager.log_metrics(metrics)

            # Log fitted scaler artifact
            with tempfile.TemporaryDirectory():
                preprocessor.save_scaler()
                path = f"{preprocessor.artifacts_dir}/scaler.pkl"
                self.manager.log_artifact(path, artifact_path=f"fold_{fold_num}/scaler")
