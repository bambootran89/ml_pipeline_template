from typing import Any

from ...eval.fold_evaluator import FoldEvaluator
from ...preprocess.fold_scaler import FoldPreprocessor
from ...tracking.cv_logger import FoldLogger
from ...trainer.dispatcher import FoldTrainer
from ...trainer.setup_helper import FoldModelBuilder
from .fold_context import FoldContext


class FoldRunner:
    """
    Orchestrates all steps required to execute a single cross-validation fold.

    This class coordinates preprocessing, model creation, training,
    evaluation, and logging by delegating work to specialized helper
    classes. It acts as the high-level controller for the fold pipeline,
    keeping the workflow clean and modular.
    """

    def __init__(self, cfg: Any, mlflow_manager):
        self.cfg = cfg
        self.preprocessor = FoldPreprocessor(cfg)
        self.model_builder = FoldModelBuilder(cfg)
        self.trainer = FoldTrainer()
        self.evaluator = FoldEvaluator()
        self.logger = FoldLogger(mlflow_manager)

    def _slice(self, x_full, y_full, train_idx, test_idx):
        """Return train and test splits for the current fold."""
        return (x_full[train_idx], y_full[train_idx]), (
            x_full[test_idx],
            y_full[test_idx],
        )

    def _prepare_context(
        self, fold_num, train_idx, test_idx, x_full, y_full, is_tuning
    ):
        """
        Construct a FoldContext containing raw data and fold metadata.
        """
        (x_train_raw, y_train), (x_test_raw, y_test) = self._slice(
            x_full, y_full, train_idx, test_idx
        )
        return FoldContext(
            fold_num,
            train_idx,
            test_idx,
            x_train_raw,
            y_train,
            x_test_raw,
            y_test,
            is_tuning,
        )

    def run_fold(
        self,
        fold_num,
        train_idx,
        test_idx,
        x_full_raw,
        y_full,
        model_name,
        hyperparams,
        is_tuning=False,
    ):
        """
        Execute the full pipeline for one fold:
        preprocessing → model/trainer setup → training → evaluation → logging.

        Parameters
        ----------
        fold_num : int
            Index of the fold.
        train_idx : Any
            Indices for training samples.
        test_idx : Any
            Indices for test/validation samples.
        x_full_raw : np.ndarray
            Full raw feature array (fold slicing done internally).
        y_full : np.ndarray
            Full target array.
        model_name : str
            Name of the model to instantiate.
        hyperparams : dict
            Hyperparameters passed to model and trainer.
        is_tuning : bool, default=False
            Whether this fold is used during hyperparameter search.

        Returns
        -------
        dict
            Evaluation metrics for the fold.
        """
        ctx = self._prepare_context(
            fold_num, train_idx, test_idx, x_full_raw, y_full, is_tuning
        )

        # Preprocessing
        ctx.preprocessor, ctx.x_train_scaled = self.preprocessor.fit(ctx.x_train_raw)
        ctx.x_test_scaled = self.preprocessor.transform(
            ctx.x_test_raw, ctx.preprocessor
        )

        # Model & trainer setup
        ctx.wrapper, ctx.trainer = self.model_builder.create(model_name, hyperparams)

        if ctx.is_tuning and hasattr(ctx.trainer, "artifacts_dir"):
            ctx.trainer.artifacts_dir = None

        # Training
        ctx.wrapper = self.trainer.train(
            ctx.wrapper,
            ctx.trainer,
            hyperparams,
            (ctx.x_train_scaled, ctx.y_train),
            (ctx.x_test_scaled, ctx.y_test),
        )

        # Evaluation
        ctx.metrics = self.evaluator.evaluate(
            ctx.wrapper, (ctx.x_test_scaled, ctx.y_test)
        )

        # Logging
        if not is_tuning:
            self.logger.log(
                ctx.fold_num,
                model_name,
                hyperparams,
                ctx.train_idx,
                ctx.test_idx,
                ctx.metrics,
                ctx.preprocessor,
            )

        return ctx.metrics
