"""
fold_runner.py

Encapsulate execution for a single cross-validation fold.

Responsibilities:
- Create model wrapper + trainer
- Build DL model when required
- Prepare dataloaders for DL
- Train model
- Evaluate and optionally log to MLflow
"""

from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader

from mlproject.src.datamodule.dataset import NumpyWindowDataset
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.trainer.dl_trainer import DeepLearningTrainer
from mlproject.src.trainer.trainer_factory import TrainerFactory


class FoldRunner:
    """Run a single cross-validation fold."""

    def __init__(self, cfg: Any, mlflow_manager: MLflowManager) -> None:
        """
        Initialize FoldRunner.

        Parameters
        ----------
        cfg : Any
            Experiment configuration (OmegaConf DictConfig).
        mlflow_manager : MLflowManager
            MLflow helper (may be a disabled stub).
        """
        self.cfg = cfg
        self.mlflow_manager = mlflow_manager
        self.evaluator = TimeSeriesEvaluator()

    def _slice_fold_data(
        self, x_full: np.ndarray, y_full: np.ndarray, train_idx: Any, test_idx: Any
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Return train and test subsets for the fold."""
        x_train, y_train = x_full[train_idx], y_full[train_idx]
        x_test, y_test = x_full[test_idx], y_full[test_idx]
        return (x_train, y_train), (x_test, y_test)

    def _prepare_dataloaders(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader]:
        """Return train and validation DataLoader for DL models."""
        train_loader = DataLoader(
            NumpyWindowDataset(x_train, y_train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            NumpyWindowDataset(x_val, y_val), batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader

    def _create_model_and_trainer(
        self, model_name: str, hyperparams: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Create model wrapper and trainer using factory helpers."""
        wrapper = ModelFactory.create(model_name, hyperparams)
        trainer = TrainerFactory.create(
            model_name, wrapper, self.cfg.training.artifacts_dir
        )
        return wrapper, trainer

    def _train_dl(
        self,
        trainer: DeepLearningTrainer,
        wrapper: Any,
        hyperparams: Dict[str, Any],
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> Any:
        """Train a deep-learning trainer and return the trained wrapper."""
        x_train, y_train = train_data
        x_test, y_test = test_data
        input_dim = x_train.shape[1] * x_train.shape[2]
        output_dim = y_train.shape[1]
        wrapper.build(input_dim, output_dim)

        batch_size = int(hyperparams.get("batch_size", 16))
        train_loader, val_loader = self._prepare_dataloaders(
            x_train, y_train, x_test, y_test, batch_size
        )
        return trainer.train(train_loader, val_loader, hyperparams)

    def _train_ml(
        self,
        trainer: Any,
        wrapper: Any,
        hyperparams: Dict[str, Any],
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> Any:
        """Train classical ML trainer and return trained wrapper."""
        assert wrapper is not None
        x_train, y_train = train_data
        x_test, y_test = test_data
        return trainer.train((x_train, y_train), (x_test, y_test), hyperparams)

    def _evaluate(
        self, wrapper: Any, test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Run prediction + evaluation and convert metrics to float."""
        x_test, y_test = test_data
        preds = wrapper.predict(x_test)
        raw_metrics = self.evaluator.evaluate(y_test, preds)
        return {k: float(v) for k, v in raw_metrics.items()}

    def _log_fold(
        self,
        fold_num: int,
        model_name: str,
        hyperparams: Dict[str, Any],
        train_idx: Any,
        test_idx: Any,
        metrics: Dict[str, float],
    ) -> None:
        """Log parameters and metrics for a fold."""
        # This method creates a run for EACH fold.
        # We only call this if NOT tuning.
        if not (self.mlflow_manager and getattr(self.mlflow_manager, "enabled", False)):
            return

        with self.mlflow_manager.start_run(
            run_name=f"{model_name}_fold_{fold_num}", nested=True
        ):
            self.mlflow_manager.log_params(
                {
                    "fold": fold_num,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **hyperparams,
                }
            )
            self.mlflow_manager.log_metrics(metrics)
            # LOGIC: Could log model artifacts here if needed for debugging single runs

    def run_fold(
        self,
        fold_num: int,
        train_idx: Any,
        test_idx: Any,
        x_full: np.ndarray,
        y_full: np.ndarray,
        model_name: str,
        hyperparams: Dict[str, Any],
        is_tuning: bool = False,  # <--- NEW PARAM
    ) -> Dict[str, float]:
        """
        Docstring for run_fold

        :param self: Description
        :param fold_num: Description
        :type fold_num: int
        :param train_idx: Description
        :type train_idx: Any
        :param test_idx: Description
        :type test_idx: Any
        :param x_full: Description
        :type x_full: np.ndarray
        :param y_full: Description
        :type y_full: np.ndarray
        :param model_name: Description
        :type model_name: str
        :param hyperparams: Description
        :type hyperparams: Dict[str, Any]
        :param is_tuning: Description
        :type is_tuning: bool
        :return: Description
        :rtype: Dict[str, float]
        """
        # 1. Slice data
        train_data, test_data = self._slice_fold_data(
            x_full, y_full, train_idx, test_idx
        )

        # 2. Create model + trainer
        wrapper, trainer = self._create_model_and_trainer(model_name, hyperparams)

        if is_tuning and hasattr(trainer, "artifacts_dir"):
            trainer.artifacts_dir = None

        # 3. Train
        if isinstance(trainer, DeepLearningTrainer):
            wrapper = self._train_dl(
                trainer, wrapper, hyperparams, train_data, test_data
            )
        else:
            wrapper = self._train_ml(
                trainer, wrapper, hyperparams, train_data, test_data
            )

        # 4. Evaluate
        metrics = self._evaluate(wrapper, test_data)

        # 5. Log metrics (ONLY if NOT tuning)
        # During tuning, we only care about the aggregated metric of the Trial,
        # not the individual folds to keep MLflow UI clean.
        if not is_tuning:
            self._log_fold(
                fold_num, model_name, hyperparams, train_idx, test_idx, metrics
            )

        return metrics
