import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ARTIFACT_DIR = os.path.join("mlproject", "artifacts", "preprocessing")


class PreprocessBase:
    """
    Base preprocessing logic used for BOTH offline training and online serving.

    This class contains all shared logic to ensure consistency between
    model training and model inference (online API).

    It implements:
    - Missing value imputation
    - Covariate generation
    - Scaler fit + save
    - Scaler load + transform
    """

    def __init__(self, cfg=None):
        """
        Initialize the preprocessing base object.

        Args:
            cfg (dict, optional):
                Configuration dictionary containing
                preprocessing steps and artifact path.
        """
        self.cfg = cfg or {}

        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])

        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.scaler = None
        self.scaler_columns = None

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing steps on the DataFrame.

        Steps include:
        - fill_missing
        - generate_covariates
        - fit scaler (normalize step)

        Args:
            df (pd.DataFrame): Input raw DataFrame.

        Returns:
            pd.DataFrame: DataFrame after preprocessing steps (scaler fitted).
        """
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        df = self._apply_fit_scaler(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to the DataFrame.

        Steps include:
        - fill_missing
        - generate_covariates
        - load scaler (if needed)
        - apply scaling

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed output.
        """
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        if self.scaler is None:
            self.load_scaler()
        df = self._apply_scaling(df)
        return df

    def _apply_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value imputation according to config.

        Supported:
        - ffill
        - mean

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
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

    def _apply_generate_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate additional covariates based on config.

        Currently supports:
        - day_of_week: requires DatetimeIndex

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with generated covariates.
        """
        step = self._get_step("gen_covariates")
        if not step:
            return df

        cov = step.get("covariates", {})

        if "future" in cov and "day_of_week" in cov["future"]:
            if isinstance(df.index, pd.DatetimeIndex):
                df["day_of_week"] = df.index.dayofweek
        return df

    def _apply_fit_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler (StandardScaler or MinMaxScaler) based on config.

        Saves fitted scaler to artifacts_dir/scaler.pkl.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame unchanged (scaler fitted separately).
        """
        step = self._get_step("normalize")
        if not step:
            return df

        cols = self._get_numeric_columns(df, step)
        method = step.get("method", "zscore")

        if method == "zscore":
            scaler = StandardScaler().fit(df[cols].values)
        elif method == "minmax":
            scaler = MinMaxScaler().fit(df[cols].values)
        else:
            raise ValueError(f"Unknown normalize method: {method}")

        self.scaler = scaler
        self.scaler_columns = cols

        self._save_scaler()

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously fitted scaler to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        if self.scaler is None:
            return df
        # Ensure all columns exist
        for c in self.scaler_columns:
            if c not in df.columns:
                df[c] = 0.0

        df[self.scaler_columns] = self.scaler.transform(df[self.scaler_columns])
        return df

    def _save_scaler(self):
        """
        Save scaler + column list to artifact directory.

        Creates file:
            artifacts/preprocessing/scaler.pkl
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        with open(os.path.join(self.artifacts_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "columns": self.scaler_columns},
                f,
            )

    def load_scaler(self):
        """
        Load scaler + column list from artifact directory if not loaded.
        """
        path = os.path.join(self.artifacts_dir, "scaler.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.scaler_columns = obj["columns"]

    def _get_step(self, name: str):
        """
        Retrieve a preprocessing step by name.

        Args:
            name (str): Step name.

        Returns:
            dict or None: Step config or None if not found.
        """
        return next((s for s in self.steps if s.get("name") == name), None)

    def _get_numeric_columns(self, df, step):
        """
        Get numeric columns to scale.

        Args:
            df (pd.DataFrame): Input DataFrame.
            step (dict): normalize step config.

        Returns:
            list[str]: Numeric column names.
        """
        cols = step.get("columns")
        if cols:
            return cols

        return df.select_dtypes(include=[np.number]).columns.tolist()
