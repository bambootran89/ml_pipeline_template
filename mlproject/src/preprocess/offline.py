# mlproject/src/preprocess/offline.py

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ARTIFACT_DIR = os.path.join("mlproject", "artifacts", "preprocessing")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class OfflinePreprocessor:
    """
    Offline data preprocessor for time series data.

    Handles missing value imputation, feature generation, scaling, and saving artifacts.

    Args:
        cfg (dict, optional): Configuration dictionary specifying preprocessing steps
                              and artifact directory.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        print("HELLO", self.steps)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing steps on the input DataFrame
        (fill missing, generate features, fit scaler).

        Args:
            df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame:
            DataFrame after fitting preprocessing steps.
        """
        df = df.copy()
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        df = self._apply_fit_scaler(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to a DataFrame and save features.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        df = df.copy()
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        df = self._apply_scaling(df)
        self._save_features(df)
        return df

    def run(self) -> pd.DataFrame:
        """
        Execute the full offline preprocessing pipeline:
        load raw data, fit preprocessing, transform, and save features.

        Returns:
            pd.DataFrame: Final processed DataFrame.
        """
        df = self._load_raw_data()
        df = self.fit(df)
        df = self.transform(df)
        return df

    def _apply_fill_missing(self, df):
        """
        Fill missing values in the DataFrame according to the configured method.

        Supports 'ffill' (forward/backward fill) and 'mean'.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.
        """
        step = self._get_step("fill_missing")
        if not step:
            return df

        method = step.get("method", "ffill")

        if method == "ffill":
            return df.fillna(method="ffill").fillna(method="bfill")

        if method == "mean":
            return df.fillna(df.mean())

        raise ValueError(f"Unknown fill_missing method: {method}")

    def _apply_generate_covariates(self, df):
        """
        Generate additional covariates/features based on configuration.

        Example: generate 'day_of_week' if index is datetime and specified in config.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with generated covariates.
        """
        step = self._get_step("gen_covariates")
        if not step:
            return df

        cov = step.get("covariates", {})

        # SAFE GUARD: Only generate feature if index is datetime
        if "future" in cov and "day_of_week" in cov["future"]:
            if isinstance(df.index, pd.DatetimeIndex):
                df["day_of_week"] = df.index.dayofweek
        return df

    def _apply_fit_scaler(self, df):
        """
        Fit a scaler (StandardScaler or MinMaxScaler) on numeric columns and save it.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame (scaler stored as artifact).
        """
        step = self._get_step("normalize")
        if not step:
            return df

        method = step.get("method", "zscore")
        cols = self._get_numeric_columns(df, step)

        scaler = self._create_scaler(method, df[cols].values)
        self._save_scaler(scaler, cols)

        return df

    def _safe_scaler_cols(self, scaler, cols):
        """
        Ensure column names from scaler are valid for assignment.

        Args:
            scaler: Fitted scaler object.
            cols (list): Original column names.

        Returns:
            list: Safe column names.
        """
        if hasattr(scaler, "feature_names_in_"):
            return list(scaler.feature_names_in_)
        return list(cols)

    def _apply_scaling(self, df):
        """
        Apply saved scaler to DataFrame for normalization.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        scaler_data = self._load_scaler()
        if not scaler_data:
            return df

        scaler = scaler_data["scaler"]
        cols = scaler_data["columns"]
        safe_cols = self._safe_scaler_cols(scaler, cols)

        df[safe_cols] = scaler.transform(df[safe_cols])

        return df

    def _save_scaler(self, scaler, cols):
        """
        Save the scaler object and its columns to artifact directory.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump({"scaler": scaler, "columns": cols}, f)

    def _load_scaler(self):
        """
        Load the saved scaler object
        and columns from artifact directory.
        Returns:
            dict or None: Dictionary with 'scaler'
            and 'columns' keys, or None if not found.
        """
        path = os.path.join(self.artifacts_dir, "scaler.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_features(self, df):
        """
        Save processed DataFrame as Parquet file in artifact directory.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        df.to_parquet(os.path.join(self.artifacts_dir, "features.parquet"))

    def _load_raw_data(self):
        """
        Load raw data from CSV or generate synthetic data if path not found.

        Returns:
            pd.DataFrame: Loaded raw data.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        index_col = data_cfg.get("index_col", "date")

        if not path or not os.path.exists(path):
            return self._load_synthetic(index_col)

        return self._load_csv(path, index_col)

    def _load_csv(self, path, index_col):
        """
        Load CSV data and set specified index column.

        Args:
            path (str): CSV file path.
            index_col (str): Column to set as index.

        Returns:
            pd.DataFrame: DataFrame with index set.
        """
        df = pd.read_csv(path, parse_dates=[index_col])
        return df.set_index(index_col)

    def _load_synthetic(self, index_col):
        """
        Generate synthetic DataFrame for testing or fallback.

        Args:
            index_col (str): Name of the index column.

        Returns:
            pd.DataFrame: Synthetic DataFrame.
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

    def _get_step(self, name):
        """
        Generate synthetic DataFrame for testing or fallback.

        Args:
            index_col (str): Name of the index column.

        Returns:
            pd.DataFrame: Synthetic DataFrame.
        """
        return next((s for s in self.steps if s.get("name") == name), None)

    def _get_numeric_columns(self, df, step):
        """
        Determine numeric columns for scaling, optionally filtered by step config.

        Args:
            df (pd.DataFrame): Input DataFrame.
            step (dict): Step configuration.

        Returns:
            list: Numeric column names.
        """
        cols = step.get("columns")
        if cols:
            return cols
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _create_scaler(self, method, values):
        """
        Create and fit a scaler (StandardScaler or MinMaxScaler) on given values.

        Args:
            method (str): 'zscore' or 'minmax'.
            values (np.ndarray): Values to fit scaler on.

        Returns:
            Scaler object: Fitted scaler.
        """
        if method == "zscore":
            return StandardScaler().fit(values)
        if method == "minmax":
            return MinMaxScaler().fit(values)
        raise ValueError(f"Unknown normalize method: {method}")
