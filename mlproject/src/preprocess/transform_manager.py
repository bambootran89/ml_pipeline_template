"""
Transform Manager.

This module defines `TransformManager`, a utility class responsible for
managing stateful preprocessing transforms, including:

- Missing value imputation (mean / median / mode)
- Label encoding for categorical features
- Feature scaling (StandardScaler / MinMaxScaler)

All fitted states can be persisted and restored from disk to ensure
reproducibility across training and inference.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# from mlproject.src.utils.func_utils import sync_artifacts_from_registry

ScalerType = Union[StandardScaler, MinMaxScaler]


class FillnaStat(TypedDict):
    """Typed structure for fillna statistics of a single column."""

    method: str
    value: float


class TransformManager:
    """
    Manage all stateful preprocessing transforms.

    This class encapsulates fitting, applying, saving, and loading
    preprocessing steps with internal state.

    Attributes
    ----------
    artifacts_dir : str
        Directory used to persist transform artifacts.
    fillna_stats : Dict[str, LabelEncoder]
        Per-column missing value statistics.
    label_encoders : Dict[str, LabelEncoder]
        Label encoders for categorical columns.
    scaler : Optional[ScalerType]
        Fitted scaler instance.
    scaler_columns : Optional[List[str]]
        Columns used when fitting the scaler.
    """

    def __init__(self, artifacts_dir: str) -> None:
        """
        Initialize the TransformManager.

        Parameters
        ----------
        artifacts_dir : str
            Directory where transform artifacts are stored.
        """
        self.artifacts_dir: str = artifacts_dir

        self.fillna_stats: Dict[str, FillnaStat] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[ScalerType] = None
        self.scaler_columns: Optional[List[str]] = None

    def fit_fillna(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "mean",
    ) -> None:
        """
        Fit missing-value statistics for the specified columns.

        The computed statistics are stored internally and later used by
        :meth:`transform_fillna` to impute missing values.

        Supported strategies:
            - "mean":   Column-wise mean
            - "median": Column-wise median
            - "mode":   Most frequent value

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame used to compute fillna statistics.
        columns : List[str]
            List of column names for which fillna statistics are computed.
        method : str, default="mean"
            Fillna strategy. Must be one of {"mean", "median", "mode"}.

        Raises
        ------
        ValueError
            If an unsupported fillna method is provided.
        """
        if method not in {"mean", "median", "mode"}:
            raise ValueError(f"Invalid fillna method: {method}")

        for col in columns:
            if col not in df.columns:
                continue

            if method == "mean":
                value: float = float(df[col].mean())
            elif method == "median":
                value = float(df[col].median())
            else:
                modes = df[col].mode()
                value = float(modes.iloc[0]) if not modes.empty else 0.0

            self.fillna_stats[col] = {
                "method": method,
                "value": value,
            }

    def transform_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value imputation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        df_out = df.copy()

        for col, stats in self.fillna_stats.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(stats["value"])

        return df_out

    def fit_label_encoding(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> None:
        """
        Fit label encoders for categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[str]
            Categorical columns to encode.
        """
        for col in columns:
            if col not in df.columns:
                continue

            encoder = LabelEncoder()
            non_null_values = df[col].dropna()

            if not non_null_values.empty:
                encoder.fit(non_null_values.astype(str))
                self.label_encoders[col] = encoder

    @staticmethod
    def _encode_series_safe(
        values: pd.Series,
        encoder: LabelEncoder,
    ) -> pd.Series:
        """
        Encode a pandas Series using a fitted LabelEncoder.

        Any unseen label is mapped to the first known class.

        Parameters
        ----------
        values : pd.Series
            Input categorical values (no NaNs).
        encoder : LabelEncoder
            Fitted label encoder.

        Returns
        -------
        pd.Series
            Encoded integer values.
        """
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]

        safe_values = values.where(values.isin(known_classes), default_class)
        return pd.Series(
            encoder.transform(safe_values),
            index=values.index,
        )

    def transform_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply label encoding to categorical columns.

        Unseen labels are mapped to the first known class.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        df_out = df.copy()

        for col, encoder in self.label_encoders.items():
            if col not in df_out.columns:
                continue

            mask = df_out[col].notna()
            if not mask.any():
                continue

            values = df_out.loc[mask, col].astype(str)

            df_out.loc[mask, col] = self._encode_series_safe(
                values=values,
                encoder=encoder,
            )

        return df_out

    def fit_scaler(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "zscore",
    ) -> None:
        """
        Fit a feature scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : List[str]
            Columns to scale.
        method : str, default="zscore"
            One of {"zscore", "minmax"}.

        Raises
        ------
        ValueError
            If an unsupported scaling method is provided.
        """
        if method == "zscore":
            scaler: ScalerType = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaling method: {method}")

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            return

        numeric_df = df[valid_cols].astype(np.float64)
        scaler.fit(numeric_df.values)

        self.scaler = scaler
        self.scaler_columns = valid_cols

    def transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature scaling.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        if self.scaler is None or self.scaler_columns is None:
            return df

        df_out = df.copy()
        valid_cols = [c for c in self.scaler_columns if c in df_out.columns]

        if not valid_cols:
            return df_out

        for col in valid_cols:
            if not pd.api.types.is_float_dtype(df[col]):
                df_out[col] = df_out[col].astype(np.float64)

        scaled_values = self.scaler.transform(
            df_out[valid_cols].to_numpy(dtype=np.float64)
        )
        df_out.loc[:, valid_cols] = scaled_values

        return df_out

    def save(self) -> None:
        """
        Persist all transform states to disk.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)

        with open(os.path.join(self.artifacts_dir, "fillna_stats.pkl"), "wb") as f:
            pickle.dump(self.fillna_stats, f)

        with open(os.path.join(self.artifacts_dir, "label_encoders.pkl"), "wb") as f:
            pickle.dump(self.label_encoders, f)

        with open(os.path.join(self.artifacts_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "columns": self.scaler_columns,
                },
                f,
            )

    def load(self, cfg) -> None:
        """
        Load all transform states from disk.
        """
        # current_cfg = cfg or {}
        # model_name = (
        #     current_cfg.get("mlflow", {})
        #     .get("registry", {})
        #     .get("model_name", "ts_forecast_model")
        # )
        # sync_artifacts_from_registry(model_name, self.artifacts_dir)
        _ = cfg
        path = os.path.join(self.artifacts_dir, "fillna_stats.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.fillna_stats = pickle.load(f)

        path = os.path.join(self.artifacts_dir, "label_encoders.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.label_encoders = pickle.load(f)

        path = os.path.join(self.artifacts_dir, "scaler.pkl")
        print(f"Loading artifacts {path}")
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
                self.scaler = obj["scaler"]
                self.scaler_columns = obj["columns"]

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve transform parameters for experiment tracking.

        Returns
        -------
        Dict[str, Any]
            Dictionary of transform parameters.
        """
        params: Dict[str, Any] = {}

        for col, stats in self.fillna_stats.items():
            params[f"fillna.{col}.method"] = stats["method"]
            params[f"fillna.{col}.value"] = stats["value"]

        for col, encoder in self.label_encoders.items():
            params[f"label_encode.{col}.n_classes"] = len(encoder.classes_)
            params[f"label_encode.{col}.classes"] = ",".join(
                map(str, encoder.classes_[:5])
            )

        if self.scaler is not None:
            params["scaler.type"] = type(self.scaler).__name__
            params["scaler.n_features"] = len(self.scaler_columns or [])

        return params
