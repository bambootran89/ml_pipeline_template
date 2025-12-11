"""
FoldRunner: Run a single cross-validation fold with isolated preprocessing.

Key changes:
1. Each fold fits its own scaler on train data.
2. Validation/test data is transformed using the fold's scaler.
3. Scaler artifacts can be logged to MLflow.
"""

import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mlproject.src.datamodule.dataset import NumpyWindowDataset
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.preprocess.base import PreprocessBase
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.trainer.dl_trainer import DeepLearningTrainer
from mlproject.src.trainer.trainer_factory import TrainerFactory


class FoldRunner:
    """
    Run a single fold for cross-validation
    using raw data and fold-specific preprocessing.
    """

    def __init__(self, cfg: Any, mlflow_manager: MLflowManager) -> None:
        """
        Initialize FoldRunner.

        Args:
            cfg: Configuration object.
            mlflow_manager: MLflow manager instance.
        """
        self.cfg = cfg
        self.mlflow_manager = mlflow_manager
        self.evaluator = TimeSeriesEvaluator()

    def _get_feature_names(self) -> list[str]:
        """Return list of feature names (from config or default)."""
        return ["HUFL", "MUFL", "mobility_inflow"]

    def _fit_fold_scaler(
        self, x_train: np.ndarray, fold_num: int
    ) -> Tuple[PreprocessBase, np.ndarray]:
        """
        Fit scaler only on train data of this fold.

        Args:
            x_train: Raw train data (n_samples, seq_len, n_features).
            fold_num: Fold index.

        Returns:
            preprocessor: Fitted PreprocessBase instance.
            x_train_scaled: Scaled train data.
        """
        preprocessor = PreprocessBase(self.cfg)
        n_samples, seq_len, n_features = x_train.shape
        x_flat = x_train.reshape(-1, n_features)
        df_train = pd.DataFrame(x_flat, columns=self._get_feature_names())
        df_train_scaled = preprocessor.fit(df_train)
        x_train_scaled = df_train_scaled.values.reshape(n_samples, seq_len, n_features)
        return preprocessor, x_train_scaled

    def _transform_with_scaler(
        self, x: np.ndarray, preprocessor: PreprocessBase
    ) -> np.ndarray:
        """
        Transform data using a fitted scaler.

        Args:
            x: Raw input data.
            preprocessor: Fitted PreprocessBase.

        Returns:
            Scaled data array.
        """
        n_samples, seq_len, n_features = x.shape
        x_flat = x.reshape(-1, n_features)
        df = pd.DataFrame(x_flat, columns=self._get_feature_names())
        df_scaled = preprocessor.transform(df)
        return df_scaled.values.reshape(n_samples, seq_len, n_features)

    def _slice_fold_data(
        self, x_full: np.ndarray, y_full: np.ndarray, train_idx: Any, test_idx: Any
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Slice raw data for a fold."""
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
        """Return train and validation DataLoaders."""
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
        """Create model wrapper and trainer instance."""
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
        """Train a deep learning model."""
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
        """Train a classical machine learning model."""
        x_train, y_train = train_data
        x_test, y_test = test_data
        return trainer.train((x_train, y_train), (x_test, y_test), hyperparams)

    def _evaluate(
        self, wrapper: Any, test_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Run prediction and compute evaluation metrics."""
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
        preprocessor: PreprocessBase,
    ) -> None:
        """Log fold metrics and scaler artifact to MLflow."""
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

            # Log scaler artifact
            with tempfile.TemporaryDirectory() as _:
                preprocessor._save_scaler()
                scaler_path = preprocessor.artifacts_dir + "/scaler.pkl"
                self.mlflow_manager.log_artifact(
                    scaler_path, artifact_path=f"fold_{fold_num}/scaler"
                )

    def run_fold(
        self,
        fold_num: int,
        train_idx: Any,
        test_idx: Any,
        x_full_raw: np.ndarray,
        y_full: np.ndarray,
        model_name: str,
        hyperparams: Dict[str, Any],
        is_tuning: bool = False,
    ) -> Dict[str, float]:
        """
        Run a single fold with isolated preprocessing.

        Args:
            fold_num: Fold index.
            train_idx: Training indices.
            test_idx: Testing indices.
            x_full_raw: Raw input windows (not normalized).
            y_full: Target windows.
            model_name: Name of the model.
            hyperparams: Model hyperparameters.
            is_tuning: Skip MLflow logging if True.

        Returns:
            Dictionary of evaluation metrics for the fold.
        """
        train_data_raw, test_data_raw = self._slice_fold_data(
            x_full_raw, y_full, train_idx, test_idx
        )
        x_train_raw, y_train = train_data_raw
        x_test_raw, y_test = test_data_raw

        preprocessor, x_train_scaled = self._fit_fold_scaler(x_train_raw, fold_num)
        x_test_scaled = self._transform_with_scaler(x_test_raw, preprocessor)

        wrapper, trainer = self._create_model_and_trainer(model_name, hyperparams)
        if is_tuning and hasattr(trainer, "artifacts_dir"):
            trainer.artifacts_dir = None

        if isinstance(trainer, DeepLearningTrainer):
            wrapper = self._train_dl(
                trainer,
                wrapper,
                hyperparams,
                (x_train_scaled, y_train),
                (x_test_scaled, y_test),
            )
        else:
            wrapper = self._train_ml(
                trainer,
                wrapper,
                hyperparams,
                (x_train_scaled, y_train),
                (x_test_scaled, y_test),
            )

        metrics = self._evaluate(wrapper, (x_test_scaled, y_test))

        if not is_tuning:
            self._log_fold(
                fold_num,
                model_name,
                hyperparams,
                train_idx,
                test_idx,
                metrics,
                preprocessor,
            )

        return metrics
