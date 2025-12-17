from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.utils.func_utils import (
    load_data_csv,
    normalize_preprocessing_steps,
    select_columns,
)

ARTIFACT_DIR = "mlproject/artifacts/preprocessing"


class OfflinePreprocessor:
    """
    Offline preprocessing orchestrator.

    Responsibilities
    ----------------
    - Load raw data
    - Select leakage-safe training subset
    - Fit preprocessing artifacts
    - Apply fitted transforms to full dataset
    """

    cfg: DictConfig
    is_train: bool
    steps: List[Dict[str, Any]]
    data_path: str
    artifacts_dir: str
    transform_manager: TransformManager

    def __init__(self, cfg: DictConfig, is_train: bool = True) -> None:
        """
        Initialize offline preprocessor.

        Parameters
        ----------
        cfg : DictConfig
            Experiment configuration.
        is_train : bool, default=True
            Training or inference mode.
        """
        self.cfg = cfg
        self.is_train = is_train

        self.steps = normalize_preprocessing_steps(cfg)

        self.data_path = cfg.data.path
        self.artifacts_dir = cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )

        self.transform_manager = TransformManager(artifacts_dir=self.artifacts_dir)
        self.transform_manager.steps = self.steps

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset from disk or generate synthetic data.

        Returns
        -------
        pd.DataFrame
            Raw dataset.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        if not path:
            raise ValueError("`data.path` must be specified")

        data_type = data_cfg.get("type", "timeseries").lower()
        index_col = data_cfg.get("index_col")

        if data_type == "timeseries" and not index_col:
            index_col = "date"
        if data_type == "tabular":
            index_col = None

        if not os.path.exists(path):
            return self._load_synthetic(index_col)

        return load_data_csv(
            path,
            index_col=index_col,
            data_type=data_type,
        )

    def select_train_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select training subset for fitting transforms.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset.

        Returns
        -------
        pd.DataFrame
            Training subset.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        split_cfg = self.cfg.preprocessing.split
        train_ratio = float(split_cfg.train)

        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"Invalid train split ratio: {train_ratio}")

        train_size = int(len(df) * train_ratio)
        if train_size <= 0:
            raise ValueError("Training subset size is zero")

        if self.cfg.data.type == "tabular":
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        return df.iloc[:train_size].copy()

    def get_select_df(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Select feature/target columns safely.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        include_target : bool, default=True
            Whether to include target column.

        Returns
        -------
        pd.DataFrame
            Selected dataframe.
        """
        return select_columns(
            self.cfg,
            df,
            include_target=include_target,
        )

    def fit_manager(self, train_df: pd.DataFrame) -> None:
        """
        Fit preprocessing artifacts using training data.

        Stateful transformations are fitted first.
        Stateless transformations are applied sequentially
        to keep the pipeline order consistent.
        """
        df_work = self.get_select_df(train_df, include_target=True)
        # =========================
        # 1. STATEFUL TRANSFORMS
        # =========================
        for step_cfg in self.steps:
            step_name = step_cfg.get("name")
            if not isinstance(step_name, str):
                raise ValueError("Each preprocessing step must define a string `name`")

            if step_name == "fill_missing":
                self.transform_manager.fit_fillna(
                    df_work,
                    columns=step_cfg.get("columns", list(df_work.columns)),
                    method=step_cfg.get("method", "mean"),
                )

            elif step_name == "label_encoding":
                self.transform_manager.fit_label_encoding(
                    df_work,
                    columns=step_cfg.get("columns", []),
                )

            elif step_name == "normalize":
                self.transform_manager.fit_scaler(
                    df_work,
                    columns=step_cfg.get("columns", []),
                    method=step_cfg.get("method", "zscore"),
                )
            elif step_name in [
                "clip",
                "log",
                "abs",
                "round",
                "binary",
                "exponential",
                "udf",
            ]:
                self.transform_manager.stateless_transform(df_work, step_cfg)

            else:
                raise ValueError(f"Unknown transform step: {step_name}")

        self.transform_manager.save()

    def transform_full_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transforms to full dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Transformed dataset.
        """
        df = self.get_select_df(df, include_target=True)
        return self.transform_manager.transform(df)

    def run(self) -> pd.DataFrame:
        """
        Execute offline preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Processed dataset.
        """
        df = self.load_raw_data()
        df = self.get_select_df(df, include_target=True)

        if self.is_train:
            train_df = self.select_train_subset(df)
            self.fit_manager(train_df)

        return self.transform_full_dataset(df)

    def log_artifacts_to_mlflow(self) -> None:
        """Deprecated compatibility hook."""
        return None

    def _load_synthetic(self, index_col: str | None) -> pd.DataFrame:
        """
        Generate synthetic fallback dataset.

        Parameters
        ----------
        index_col : str or None
            Index column name.

        Returns
        -------
        pd.DataFrame
            Synthetic dataset.
        """
        idx = pd.date_range(
            "2020-01-01",
            periods=200,
            freq="H",
        )
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
