"""
Minimal patch cho OfflinePreprocessor - chỉ thêm MLflow support.

Thay đổi:
1. __init__ nhận thêm mlflow_manager (optional)
2. Thêm method _log_preprocessing_artifacts
3. Gọi logging sau khi transform
"""

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from .base import ARTIFACT_DIR
from .engine import PreprocessEngine


class OfflinePreprocessor:
    """
    Offline data preprocessor for time series data.

    Handles missing value imputation, feature generation, scaling, and saving artifacts.

    Args:
        cfg (dict, optional): Configuration dictionary specifying preprocessing steps
                              and artifact directory.
        mlflow_manager (MLflowManager, optional): MLflow manager for artifact logging.
    """

    def __init__(self, cfg=None, mlflow_manager: Optional = None):  # ✅ THÊM PARAMETER
        """
        Initialize OfflinePreprocessor.

        Args:
            cfg (dict, optional): Preprocessing configuration.
            mlflow_manager (MLflowManager, optional): MLflow manager instance.
        """
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.engine = PreprocessEngine.instance(cfg)
        self.mlflow_manager = mlflow_manager  # ✅ THÊM ATTRIBUTE

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing pipeline on the DataFrame.

        Args:
            df (pd.DataFrame): Raw time series dataset.

        Returns:
            pd.DataFrame: Dataset after applying fitting steps.
        """
        return self.engine.offline_fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted preprocessing.

        Args:
            df (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        df = self.engine.offline_transform(df)
        return df

    def run(self) -> pd.DataFrame:
        """
        Execute full offline preprocessing pipeline:
        - load raw dataset
        - fit preprocessing
        - transform dataset
        - save features
        - log to MLflow (if enabled)  # ✅ THÊM

        Returns:
            pd.DataFrame: Fully processed dataset.
        """
        df = self.load_raw_data()
        df = self.fit(df)
        df = self.transform(df)

        # ✅ THÊM: Log artifacts vào MLflow
        self._log_preprocessing_artifacts(df)

        return df

    def _log_preprocessing_artifacts(self, df: pd.DataFrame):
        """
        Log preprocessing artifacts với detailed error handling.

        Args:
            df: Processed DataFrame
        """
        print("\n[Preprocessing] Starting artifact logging...")

        # Check 1: MLflow manager exists
        if not self.mlflow_manager:
            print("[Preprocessing] ⚠️  No MLflowManager provided - skipping")
            return

        # Check 2: MLflow enabled
        if not getattr(self.mlflow_manager, "enabled", False):
            print("[Preprocessing] ⚠️  MLflow disabled in config - skipping")
            return

        # Check 3: Active run exists
        try:
            import mlflow

            active_run = mlflow.active_run()

            if not active_run:
                print("[Preprocessing] ⚠️  No active MLflow run - skipping")
                print("    Hint: Make sure preprocess() is called INSIDE a mlflow run")
                return

            run_id = active_run.info.run_id
            print(f"[Preprocessing] ✅ Active run: {run_id}")

        except Exception as e:
            print(f"[Preprocessing] ❌ Error checking active run: {e}")
            return

        # Log artifacts
        success_count = 0

        try:
            # 1. Log scaler
            scaler_path = os.path.join(self.artifacts_dir, "scaler.pkl")

            if not os.path.exists(scaler_path):
                print(f"[Preprocessing] ⚠️  Scaler not found at: {scaler_path}")
            else:
                print(f"[Preprocessing] Logging scaler from: {scaler_path}")
                self.mlflow_manager.log_artifact(
                    scaler_path, artifact_path="preprocessing/scaler"
                )
                print("[Preprocessing] ✅ Scaler logged")
                success_count += 1

        except Exception as e:
            print(f"[Preprocessing] ❌ Failed to log scaler: {e}")
            import traceback

            traceback.print_exc()

        try:
            # 2. Log config
            preprocess_config = {
                "steps": self.steps,
                "artifacts_dir": self.artifacts_dir,
                "feature_names": df.columns.tolist(),
                "n_samples": len(df),
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                import yaml

                yaml.dump(preprocess_config, f)
                temp_path = f.name

            self.mlflow_manager.log_artifact(
                temp_path, artifact_path="preprocessing/config"
            )
            os.unlink(temp_path)
            print("[Preprocessing] ✅ Config logged")
            success_count += 1

        except Exception as e:
            print(f"[Preprocessing] ❌ Failed to log config: {e}")

        try:
            # 3. Log statistics
            stats = {
                "mean": {k: float(v) for k, v in df.mean().to_dict().items()},
                "std": {k: float(v) for k, v in df.std().to_dict().items()},
                "min": {k: float(v) for k, v in df.min().to_dict().items()},
                "max": {k: float(v) for k, v in df.max().to_dict().items()},
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                import json

                json.dump(stats, f, indent=2)
                temp_path = f.name

            self.mlflow_manager.log_artifact(
                temp_path, artifact_path="preprocessing/statistics"
            )
            os.unlink(temp_path)
            print("[Preprocessing] ✅ Statistics logged")
            success_count += 1

        except Exception as e:
            print(f"[Preprocessing] ❌ Failed to log statistics: {e}")

        try:
            # 4. Log params
            self.mlflow_manager.log_params(
                {
                    "preprocessing.n_features": len(df.columns),
                    "preprocessing.n_samples": len(df),
                    "preprocessing.n_steps": len(self.steps),
                }
            )
            print("[Preprocessing] ✅ Params logged")
            success_count += 1

        except Exception as e:
            print(f"[Preprocessing] ❌ Failed to log params: {e}")

        # Summary
        print(f"\n[Preprocessing] Logged {success_count}/4 artifact groups")

        if success_count == 0:
            print("[Preprocessing] ⚠️  WARNING: NO artifacts were logged!")
            print("    This will cause eval pipeline to use LOCAL scaler")

    def _save_features(self, df):
        """
        Save transformed features to artifacts directory as Parquet.

        Args:
            df (pd.DataFrame): Processed dataset.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        df.to_parquet(os.path.join(self.artifacts_dir, "features.parquet"))

    def load_raw_data(self):
        """
        Load raw dataset from CSV path provided in cfg.

        If the CSV is missing, generate synthetic time series.

        Returns:
            pd.DataFrame: Raw dataset.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        index_col = data_cfg.get("index_col", "date")

        if not path or not os.path.exists(path):
            return self._load_synthetic(index_col)

        return self._load_csv(path, index_col)

    def _load_csv(self, path, index_col):
        """
        Load CSV file into DataFrame.

        Args:
            path (str): Path to CSV file.
            index_col (str): Column to treat as datetime index.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        df = pd.read_csv(path, parse_dates=[index_col])
        return df.set_index(index_col)

    def _load_synthetic(self, index_col):
        """
        Generate synthetic data when raw CSV is unavailable.

        Args:
            index_col (str): Name of index column.

        Returns:
            pd.DataFrame: Synthetic dataset.
        """
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
