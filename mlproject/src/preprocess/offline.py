from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.utils.func_utils import load_raw_data, resolve_feature_target_columns

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

        self.artifacts_dir = cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )

        self.transform_manager = TransformManager(
            self.cfg, artifacts_dir=self.artifacts_dir
        )
        self.steps = self.transform_manager.steps

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

    def fit_manager(self, df: pd.DataFrame) -> None:
        """
        Fit preprocessing artifacts using training data.

        Stateful transformations are fitted first.
        Stateless transformations are applied sequentially
        to keep the pipeline order consistent.
        """

        df_work = resolve_feature_target_columns(self.cfg, df, include_target=True)
        if "dataset" in df.columns:
            df_work["dataset"] = df["dataset"]
        # 1. STATEFUL TRANSFORMS
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
        this_df = resolve_feature_target_columns(self.cfg, df, include_target=True)
        if "dataset" in df.columns:
            this_df["dataset"] = df["dataset"]
        this_df = self.transform_manager.transform(this_df)
        return this_df

    def fit_and_transform(self) -> pd.DataFrame:
        """
        Execute offline preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Processed dataset.
        """
        df, train_df, val_df, test_df = load_raw_data(self.cfg)
        df = resolve_feature_target_columns(self.cfg, df, include_target=True)
        train_df = resolve_feature_target_columns(
            self.cfg, train_df, include_target=True
        )
        val_df = resolve_feature_target_columns(self.cfg, val_df, include_target=True)
        test_df = resolve_feature_target_columns(self.cfg, test_df, include_target=True)

        if len(train_df) == 0:
            train_df = self.select_train_subset(df)
        self.fit_manager(train_df)
        if len(df) > 0:
            return self.transform(df)
        else:
            train_df = self.transform(train_df)
            train_df["dataset"] = "train"
            val_df = self.transform(val_df)
            val_df["dataset"] = "val"
            test_df = self.transform(test_df)
            test_df["dataset"] = "test"
            df = pd.concat([train_df, val_df, test_df], axis=0)
            return df
