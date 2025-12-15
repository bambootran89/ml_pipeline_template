from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.utils.func_utils import load_data_csv

ARTIFACT_DIR = "mlproject/artifacts/preprocessing"


class OfflinePreprocessor:
    """
    Offline preprocessing orchestrator for training pipelines.

    Responsibilities:
    - Loading raw data from disk
    - Selecting a leakage-safe training subset
    - Fitting preprocessing artifacts via TransformManager
    - Applying fitted transformations to the full dataset

    Design principles:
    - No preprocessing logic lives here
    - All transformations are delegated to TransformManager
    - Training and inference behavior is deterministic and reproducible
    """

    cfg: DictConfig
    is_train: bool
    steps: Dict[str, Dict[str, Any]]
    data_path: str
    artifacts_dir: str
    transform_manager: TransformManager

    def __init__(self, cfg: DictConfig, is_train: bool = True) -> None:
        """
        Initialize the offline preprocessor.

        Parameters
        ----------
        cfg : DictConfig
            Global experiment configuration.
        is_train : bool, default=True
            Whether the pipeline is running in training mode.
        """
        self.cfg = cfg
        self.is_train = is_train
        self.steps = {
            item["name"]: item
            for item in self.cfg.get("preprocessing", {}).get("steps", [])
        }
        self.data_path = cfg.data.path
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.transform_manager = TransformManager(artifacts_dir=self.artifacts_dir)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset from CSV or generate synthetic data.

        Returns
        -------
        pd.DataFrame
            Raw dataset with datetime index.
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

        if not os.path.exists(path):
            return self._load_synthetic(index_col)
        return load_data_csv(path, index_col=index_col, data_type=data_type)

    def select_train_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the training subset for fitting preprocessing artifacts.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset ordered by time.

        Returns
        -------
        pd.DataFrame
            Training subset.

        Raises
        ------
        ValueError
            If training subset is invalid.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        split_cfg = self.cfg.preprocessing.split
        train_ratio: float = float(split_cfg.train)
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"Invalid train split ratio: {train_ratio}")

        train_size = int(len(df) * train_ratio)
        if train_size <= 0:
            raise ValueError("Training subset size is zero.")

        if self.cfg.data.type == "tabular":
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            return df_shuffled.iloc[:train_size].copy()
        return df.iloc[:train_size]

    def fit_manager(self, train_df: pd.DataFrame) -> None:
        """
        Fit all preprocessing components using training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training subset.
        """
        print("[OfflinePreprocessor] Fitting preprocessing artifacts")

        fillna_method: str = self.steps.get("fill_missing", {}).get("method", "mean")
        default_cols = list(train_df.columns)
        fillna_columns: List[str] = self.steps.get("fill_missing", {}).get(
            "columns", default_cols
        )
        self.transform_manager.fit_fillna(
            train_df, columns=fillna_columns, method=fillna_method
        )

        categorical_cols: List[str] = self.steps.get("label_encoding", {}).get(
            "columns", []
        )
        if categorical_cols:
            self.transform_manager.fit_label_encoding(
                train_df, columns=categorical_cols
            )

        numerical_cols: List[str] = self.steps.get("normalize", {}).get("columns", [])
        if not numerical_cols:
            numerical_cols = train_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        scaler_method: str = self.steps.get("normalize", {}).get("method", "zscore")
        self.transform_manager.fit_scaler(
            train_df, columns=numerical_cols, method=scaler_method
        )

        self.transform_manager.save()
        print("[OfflinePreprocessor] Preprocessing artifacts saved")

    def transform_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing pipeline to full dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Transformed dataset.
        """
        print("[OfflinePreprocessor] Applying preprocessing transforms")
        return self.transform_manager.transform(df)

    def run(self) -> pd.DataFrame:
        """
        Execute the full offline preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Fully processed dataset.
        """
        df = self.load_raw_data()
        if self.is_train:
            train_df = self.select_train_subset(df)
            self.fit_manager(train_df)
        return self.transform_full_dataset(df)

    def log_artifacts_to_mlflow(self) -> None:
        """
        Deprecated hook for compatibility.

        Preprocessing artifacts are logged via MLflow PyFunc in training pipeline.
        """
        return None

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
