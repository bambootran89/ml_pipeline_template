"""Base preprocessing module for transformation-only logic.

This module handles:
- Missing value imputation (mean/median/mode)
- Label encoding cho categorical
- Covariate generation
- Scaling (fit + transform)

The actual persistence is delegated to `TransformManager`.
"""

import numpy as np
import pandas as pd

from mlproject.src.preprocess.transform_manager import TransformManager

ARTIFACT_DIR = "mlproject/artifacts/preprocessing"


class PreprocessBase:
    """Base class implementing core preprocessing transformations.

    This class applies preprocessing steps:
    - Filling missing values (mean/median/mode)
    - Label encoding for categorical features
    - Generating covariates
    - Scaling numerical features

    Notes:
        - Artifact handling is delegated to :class:`TransformManager`.
        - Supports stateful transforms with proper save/load.
    """

    def __init__(self, cfg=None):
        """Initialize preprocessing with configuration.

        Args:
            cfg (dict, optional): Experiment configuration containing
                preprocessing settings. Defaults to empty dict.
        """
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])

        artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )

        # Unified transform manager
        self.transform_manager = TransformManager(artifacts_dir)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing steps to the data.

        This method performs:
        1. Missing-value filling (stateful)
        2. Label encoding (stateful)
        4. Fitting scaler (stateful)

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe (passthrough for fit steps).
        """
        df = self._apply_fill_missing(df, is_fit=True)
        df = self._apply_label_encoding(df, is_fit=True)
        df = self._apply_fit_scaler(df)
        self.save()
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing steps.

        This includes:
        - Filling missing values
        - Label encoding
        - Covariate generation
        - Scaling using fitted transforms

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        df = self._apply_fill_missing(df, is_fit=False)
        df = self._apply_label_encoding(df, is_fit=False)
        df = self._apply_scaling(df)
        return df

    def _apply_fill_missing(
        self, df: pd.DataFrame, is_fit: bool = False
    ) -> pd.DataFrame:
        """Fill missing values depending on configuration.

        Supported methods:
            - `"ffill"`: forward-fill + backward-fill (fitless)
            - `"mean"`: fill with per-column mean (stateful)
            - `"median"`: fill with per-column median (stateful)
            - `"mode"`: fill with per-column mode (stateful)

        Args:
            df (pd.DataFrame): Input dataframe.
            is_fit (bool): Whether to fit statistics

        Returns:
            pd.DataFrame: Filled dataframe.

        Raises:
            ValueError: If unknown fill method is provided.
        """
        step = self._get_step("fill_missing")
        if not step:
            return df

        method = step.get("method", "ffill")

        # Fitless method
        if method == "ffill":
            return df.ffill().bfill()

        # Stateful methods
        if method in ["mean", "median", "mode"]:
            columns = step.get("columns")
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if is_fit:
                self.transform_manager.fit_fillna(df, columns, method)

            return self.transform_manager.transform_fillna(df)

        raise ValueError(f"Unknown fill_missing method: {method}")

    def _apply_label_encoding(
        self, df: pd.DataFrame, is_fit: bool = False
    ) -> pd.DataFrame:
        """Label encoding cho categorical columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            is_fit (bool): Whether to fit encoders

        Returns:
            pd.DataFrame: Encoded dataframe.
        """
        step = self._get_step("label_encoding")
        if not step:
            return df

        columns = step.get("columns", [])
        if not columns:
            return df

        if is_fit:
            self.transform_manager.fit_label_encoding(df, columns)

        return self.transform_manager.transform_label_encoding(df)

    def _apply_fit_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and persist it using TransformManager.

        Supported scaling methods:
            - `"zscore"`: StandardScaler
            - `"minmax"`: MinMaxScaler

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Unmodified dataframe (scaler is saved, not applied).

        Raises:
            ValueError: If scaling method is unknown.
        """
        step = self._get_step("normalize")
        if not step:
            return df

        cols = self._get_numeric_columns(df, step)
        method = step.get("method", "zscore")

        self.transform_manager.fit_scaler(df, cols, method)
        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        return self.transform_manager.transform_scaler(df)

    def _get_step(self, name: str):
        """Retrieve a preprocessing step by name.

        Args:
            name (str): Step name.

        Returns:
            dict or None: Step config if found, else None.
        """
        return next((s for s in self.steps if s.get("name") == name), None)

    def _get_numeric_columns(self, df: pd.DataFrame, step: dict):
        """Get numeric columns to scale based on configuration.

        Args:
            df (pd.DataFrame): Input dataframe.
            step (dict): Normalization config block.

        Returns:
            list[str]: Numerical feature names.
        """
        cols = step.get("columns")
        if cols:
            return cols
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def save(self):
        """Save all transforms via TransformManager."""
        self.transform_manager.save()

    def load(self):
        """Load all transforms via TransformManager."""
        self.transform_manager.load(self.cfg)

    def get_params(self):
        """Get params for MLflow logging."""
        return self.transform_manager.get_params()

    @property
    def artifacts_dir(self):
        """Backward-compatible artifacts directory."""
        return self.transform_manager.artifacts_dir
