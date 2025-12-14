"""Base preprocessing module for transformation-only logic.

This module handles:
- Missing value imputation
- Covariate generation
- Scaling (fit + transform)

The actual persistence (save/load) of scalers is delegated to
`ScalerManager`.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mlproject.src.preprocess.scaler_manager import ScalerManager

ARTIFACT_DIR = "mlproject/artifacts/preprocessing"


class PreprocessBase:
    """Base class implementing core preprocessing transformations.

    This class applies preprocessing steps such as:
    - Filling missing values
    - Generating covariates
    - Scaling numerical features

    Notes:
        - Artifact handling (saving/loading scalers) is delegated to
          :class:`ScalerManager`.
        - Only transform logic is implemented here.
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

        # Dedicated manager for artifact I/O
        self.scaler_manager = ScalerManager(artifacts_dir, self.cfg)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing steps to the data.

        This method performs:
        1. Missing-value filling (fitless)
        2. Covariate generation (fitless)
        3. Fitting scaler (stateful)

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe (passthrough for fit steps).
        """
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        df = self._apply_fit_scaler(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing steps.

        This includes:
        - Filling missing values
        - Covariate generation
        - Scaling using previously fitted scaler

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)

        # Load scaler on demand
        if self.scaler_manager.scaler is None:
            self.scaler_manager.load()
        df = self._apply_scaling(df)
        return df

    def _apply_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values depending on configuration.

        Supported methods:
            - `"ffill"`: forward-fill + backward-fill
            - `"mean"`: fill with per-column mean

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Filled dataframe.

        Raises:
            ValueError: If unknown fill method is provided.
        """
        step = self._get_step("fill_missing")
        if not step:
            return df

        method = step.get("method", "ffill")

        if method == "ffill":
            return df.ffill().bfill()
        if method == "mean":
            return df.fillna(df.mean())

        raise ValueError(f"Unknown fill_missing method: {method}")

    def _apply_generate_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional covariate features.

        Supported covariates include:
            - `"day_of_week"` (requires DatetimeIndex)

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with new covariates added.
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
        """Fit scaler and persist it using ScalerManager.

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

        if method == "zscore":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalize method: {method}")

        # Fit scaler
        scaler.fit(df[cols].values)

        # Assign feature names (optional for sklearn warnings)
        try:
            scaler.feature_names_in_ = np.array(cols)
        except Exception:
            pass

        # Save through ScalerManager
        self.scaler_manager.save(scaler, cols)

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        scaler = self.scaler_manager.scaler
        cols = self.scaler_manager.scaler_columns

        if scaler is None or cols is None:
            return df

        # Ensure columns exist in transform dataset
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        for col in cols:
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float64)
        df.loc[:, cols] = scaler.transform(df[cols])
        return df

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

    def save_scaler(self):
        """DEPRECATED: Use ScalerManager.save() instead."""

    def load_scaler(self):
        """DEPRECATED: Use ScalerManager.load() instead."""
        self.scaler_manager.load()

    @property
    def scaler(self):
        """Backward-compatible scaler property."""
        return self.scaler_manager.scaler

    @property
    def scaler_columns(self):
        """Backward-compatible scaler column list property."""
        return self.scaler_manager.scaler_columns

    @property
    def artifacts_dir(self):
        """Backward-compatible artifacts directory."""
        return self.scaler_manager.artifacts_dir
