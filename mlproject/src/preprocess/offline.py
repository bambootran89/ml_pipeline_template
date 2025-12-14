"""
Offline Preprocessor với enhanced MLflow logging.

Improvements:
- Log fillna parameters (mean/median/mode values)
- Log label encoding artifacts
- Log transform parameters to MLflow
"""

import json
import logging
import os
import tempfile
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml

from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.func_utils import load_data_csv

from .base import ARTIFACT_DIR
from .engine import PreprocessEngine

logger = logging.getLogger(__name__)


class OfflinePreprocessor:
    """
    Offline preprocessor for batch data.

    Handles:
    - Missing value imputation (stateful)
    - Label encoding (stateful)
    - Feature generation
    - Scaling (stateful)
    - Artifact saving and MLflow logging
    """

    def __init__(
        self,
        is_train,
        cfg: Optional[Any] = None,
        mlflow_manager: Optional[MLflowManager] = None,
    ):
        """
        Initialize OfflinePreprocessor.

        Args:
            is_train: is to load or train transforms
            cfg (dict or DictConfig, optional): Preprocessing configuration.
            mlflow_manager (MLflowManager, optional): MLflow manager instance.
        """
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.engine = PreprocessEngine(is_train, cfg)
        self.mlflow_manager = mlflow_manager

    @property
    def transform_manager(self) -> Optional[TransformManager]:
        """
        Expose the underlying TransformManager instance.

        This property provides controlled access to the TransformManager
        located in the lower layers of the pipeline hierarchy
        (Engine -> Base), allowing higher-level components such as
        TrainingPipeline to log preprocessing models or artifacts.

        Returns
        -------
        Optional[TransformManager]
            The TransformManager instance if available; otherwise, ``None``.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            return None

        base = getattr(engine, "base", None)
        if base is None:
            return None

        return getattr(base, "transform_manager", None)

    def run(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Execute full offline preprocessing pipeline.

        Steps:
        1. Load raw data.
        2. Fit preprocessing transformations.
        3. Transform dataset.

        Returns:
            Transformed DataFrame.
        """
        if df is None:
            df = self.load_raw_data()
        data_cfg = self.cfg.get("data", {})
        data_type = data_cfg.get("type", "timeseries").lower()

        fea_df = self.engine.offline_fit(df)
        fea_df = self.engine.offline_transform(fea_df)

        if data_type == "timeseries":
            return fea_df
        else:
            target_cols = data_cfg.get("target_columns", [])
            tar_df = df[target_cols]
            return pd.concat([fea_df, tar_df], axis=1)

    def log_artifacts_to_mlflow(self, df: Optional[pd.DataFrame] = None):
        """
        Log preprocessing artifacts to an active MLflow run.

        Logs:
        - Transform artifacts (fillna_stats, label_encoders, scaler)
        - Transform parameters
        - Config
        - Statistics

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

        self._log_transform_artifacts()
        self._log_transform_params()
        self._log_config(df)
        self._log_statistics(df)

    def _log_transform_artifacts(self):
        """Log all transform artifacts to MLflow."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return

        artifact_files = ["fillna_stats.pkl", "label_encoders.pkl", "scaler.pkl"]

        for filename in artifact_files:
            path = os.path.join(self.artifacts_dir, filename)
            if os.path.exists(path):
                try:
                    # Log vào subdirectory preprocessing/
                    self.mlflow_manager.log_artifact(
                        path,
                        artifact_path=f"preprocessing/{filename.replace('.pkl', '')}",
                    )
                    print(f"[OfflinePreprocessor] Logged {filename} to MLflow")
                except Exception as e:
                    print(f"[OfflinePreprocessor] Failed to log {filename}: {e}")

    def _log_transform_params(self):
        """Log transform parameters to MLflow."""
        if not self.mlflow_manager or not self.mlflow_manager.enabled:
            return

        try:
            # Get params from TransformManager
            params = self.engine.base.get_params()

            if params:
                self.mlflow_manager.log_params(params)
                print(
                    f"[OfflinePreprocessor] Logged {len(params)} transform parameters"
                )
        except Exception as e:
            print(f"[OfflinePreprocessor] Failed to log params: {e}")

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

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset from CSV or generate synthetic data.

        Returns:
            Raw DataFrame with datetime index.
        """
        data_cfg = self.cfg.get("data", {})

        path = data_cfg.get("path")
        if not path:
            raise ValueError("`data.path` must be specified in config.")

        data_type = data_cfg.get("type", "timeseries").lower()

        index_col = data_cfg.get("index_col")

        if data_type == "timeseries" and not index_col:
            index_col = "date"

        if data_type == "tabular":
            index_col = None

        if not path or not os.path.exists(path):
            return self._load_synthetic(index_col)
        return load_data_csv(path, index_col, data_type)

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
