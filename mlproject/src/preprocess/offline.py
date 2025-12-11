import json
import os
import tempfile
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml

from mlproject.src.tracking.mlflow_manager import MLflowManager

from .base import ARTIFACT_DIR
from .engine import PreprocessEngine


class OfflinePreprocessor:
    """
    Offline preprocessor for time series data.

    Handles missing value imputation, feature generation, scaling,
    and saving preprocessing artifacts. Designed for offline batch
    preprocessing, separate from MLflow logging.
    """

    def __init__(
        self, cfg: Optional[Any] = None, mlflow_manager: Optional[MLflowManager] = None
    ):
        """
        Initialize OfflinePreprocessor.

        Args:
            cfg (dict or DictConfig, optional): Preprocessing configuration.
            mlflow_manager (MLflowManager, optional): MLflow manager instance.
        """
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.engine = PreprocessEngine.instance(cfg)
        self.mlflow_manager = mlflow_manager

    def run(self) -> pd.DataFrame:
        """
        Execute full offline preprocessing pipeline.

        Steps:
        1. Load raw data.
        2. Fit preprocessing transformations.
        3. Transform dataset.

        Returns:
            Transformed DataFrame.
        """
        df = self.load_raw_data()
        df = self.engine.offline_fit(df)
        df = self.engine.offline_transform(df)
        return df

    def log_artifacts_to_mlflow(self, df: Optional[pd.DataFrame] = None):
        """
        Log preprocessing artifacts to an active MLflow run.

        Args:
            df: Optional transformed DataFrame used for logging statistics.

        Note:
            Must be called inside a 'with mlflow.start_run():' block.
        """
        if not self.mlflow_manager or not getattr(
            self.mlflow_manager, "enabled", False
        ):
            return

        if not mlflow.active_run():
            return

        self._log_scaler()
        self._log_config(df)
        self._log_statistics(df)
        self._log_params(df)

    def _log_scaler(self):
        """Log the fitted scaler artifact."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return

        scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                # Log artifact into a subdirectory 'preprocessing/scaler'
                self.mlflow_manager.log_artifact(
                    scaler_path, artifact_path="preprocessing/scaler"
                )
                print(
                    "[OfflinePreprocessor] Logged scaler to MLflow: \
                        preprocessing/scaler/scaler.pkl"
                )
            except Exception as e:
                print(f"[OfflinePreprocessor] Failed to log scaler: {e}")

    def _log_config(self, df: Optional[pd.DataFrame] = None):
        """Log preprocessing configuration to MLflow."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return

        preprocess_config = {
            "steps": self.steps,
            "artifacts_dir": self.artifacts_dir,
        }
        if df is not None:
            preprocess_config["feature_names"] = df.columns.tolist()
            preprocess_config["n_samples"] = len(df)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(preprocess_config, f)
                temp_path = f.name

            self.mlflow_manager.log_artifact(
                temp_path, artifact_path="preprocessing/config"
            )
            os.unlink(temp_path)
        except Exception:
            pass

    def _log_statistics(self, df: Optional[pd.DataFrame] = None):
        """Log statistics of the DataFrame if provided."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return
        if df is None:
            return

        stats = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(stats, f, indent=2)
                temp_path = f.name

            self.mlflow_manager.log_artifact(
                temp_path, artifact_path="preprocessing/statistics"
            )
            os.unlink(temp_path)
        except Exception:
            pass

    def _log_params(self, df: Optional[pd.DataFrame] = None):
        """Log preprocessing parameters to MLflow."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return

        try:
            params = {"preprocessing.n_steps": len(self.steps)}
            if df is not None:
                params.update(
                    {
                        "preprocessing.n_features": len(df.columns),
                        "preprocessing.n_samples": len(df),
                    }
                )
            self.mlflow_manager.log_params(params)
        except Exception:
            pass

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset from CSV or generate synthetic data.

        Returns:
            Raw DataFrame with datetime index.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        index_col = data_cfg.get("index_col", "date")

        if not path or not os.path.exists(path):
            return self._load_synthetic(index_col)
        return self._load_csv(path, index_col)

    def _load_csv(self, path: str, index_col: str) -> pd.DataFrame:
        """Load CSV dataset."""
        df = pd.read_csv(path, parse_dates=[index_col])
        return df.set_index(index_col)

    def _load_synthetic(self, index_col: str) -> pd.DataFrame:
        """Generate synthetic dataset."""
        idx = pd.date_range("2020-01-01", periods=200, freq="H")
        df = pd.DataFrame(
            {
                "HUFL": np.sin(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "MUFL": np.cos(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "mobility_inflow": np.random.rand(len(idx)) * 10,
            },
            index=idx,
        )
        df.index.name = index_col
        return df
