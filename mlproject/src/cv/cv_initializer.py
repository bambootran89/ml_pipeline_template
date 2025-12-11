"""
CVInitializer: Prepare raw data for cross-validation without preprocessing leakage.

Key points:
1. Load raw CSV data
2. Apply windowing only (no normalization)
3. Each fold fits its own scaler independently
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.cv.splitter import TimeSeriesSplitter


class CVInitializer:
    """
    Prepare cross-validation context using raw data (no preprocessing applied).
    """

    def __init__(self, cfg: DictConfig, splitter: TimeSeriesSplitter) -> None:
        """
        Initialize CVInitializer with configuration and a time series splitter.

        Args:
            cfg: Configuration object containing data paths and parameters.
            splitter: TimeSeriesSplitter instance defining CV splits.
        """
        self.cfg = cfg
        self.splitter = splitter

    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw CSV data without any preprocessing.

        Returns:
            DataFrame: Raw data with datetime index.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        index_col = data_cfg.get("index_col", "date")

        if not path:
            raise ValueError("Data path not specified in config")

        df = pd.read_csv(path, parse_dates=[index_col])
        df = df.set_index(index_col)
        return df

    def _create_windows_raw(
        self, df: pd.DataFrame, input_chunk: int, output_chunk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create windowed data from raw DataFrame without normalization.

        Args:
            df: Raw DataFrame.
            input_chunk: Number of input time steps.
            output_chunk: Number of output time steps.

        Returns:
            x_windows: Input windows (n_samples, input_chunk, n_features).
            y_windows: Target windows (n_samples, output_chunk).
        """
        target_col = self.cfg.data.get("target_columns", ["HUFL"])[0]
        n = len(df)
        x_windows, y_windows = [], []

        for end_idx in range(input_chunk, n - output_chunk + 1):
            start_idx = end_idx - input_chunk
            x_window = df.iloc[start_idx:end_idx].values
            y_window = df.iloc[end_idx : end_idx + output_chunk][target_col].values
            x_windows.append(x_window)
            y_windows.append(y_window)

        if len(x_windows) == 0:
            n_features = len(df.columns)
            return (np.zeros((0, input_chunk, n_features)), np.zeros((0, output_chunk)))

        return np.stack(x_windows), np.stack(y_windows)

    def _get_total_folds(self) -> int:
        """
        Get the total number of cross-validation folds.

        Returns:
            int: Number of folds or -1 if not defined in splitter.
        """
        return getattr(self.splitter, "n_splits", -1)

    def initialize(
        self, approach: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any], int]:
        """
        Prepare raw input/output windows and model info for cross-validation.

        Args:
            approach: Dictionary with model configuration and hyperparameters.

        Returns:
            x_full_raw: Raw input windows (not normalized).
            y_full: Target windows.
            model_name: Name of the model.
            hyperparams: Model hyperparameters.
            total_folds: Number of CV folds.
        """
        # 1. Load raw data
        df_raw = self._load_raw_data()

        # 2. Create windows from raw data
        input_chunk = approach.get("hyperparams", {}).get("input_chunk_length", 24)
        output_chunk = approach.get("hyperparams", {}).get("output_chunk_length", 6)
        x_full_raw, y_full = self._create_windows_raw(df_raw, input_chunk, output_chunk)

        # 3. Extract model info
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})
        total_folds = self._get_total_folds()

        print(f"[CVInitializer] Loaded raw data shape: {x_full_raw.shape}")
        print("[CVInitializer] Data is NOT normalized; each fold fits its own scaler")

        return x_full_raw, y_full, model_name, hyperparams, total_folds
