"""
Fixed CVInitializer: Trả về RAW data thay vì preprocessed data.

Key changes:
1. Load raw CSV data
2. Chỉ apply windowing (không normalize)
3. Mỗi fold tự fit scaler riêng
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.cv.splitter import TimeSeriesSplitter


class CVInitializer:
    """Prepare CV context with RAW data (no preprocessing leakage)."""

    def __init__(self, cfg: DictConfig, splitter: TimeSeriesSplitter) -> None:
        self.cfg = cfg
        self.splitter = splitter

    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw CSV data without any preprocessing.

        Returns:
            df: Raw DataFrame with datetime index
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
        Create windowed data WITHOUT normalization.

        Args:
            df: Raw DataFrame
            input_chunk: Input sequence length
            output_chunk: Output sequence length

        Returns:
            x_windows: Raw input windows (n_samples, seq_len, n_features)
            y_windows: Target windows (n_samples, horizon)
        """
        target_col = self.cfg.data.get("target_columns", ["HUFL"])[0]

        n = len(df)
        x_windows, y_windows = [], []

        for end_idx in range(input_chunk, n - output_chunk + 1):
            start_idx = end_idx - input_chunk

            # ✅ RAW values (không normalize)
            x_window = df.iloc[start_idx:end_idx].values
            y_window = df.iloc[end_idx : end_idx + output_chunk][target_col].values

            x_windows.append(x_window)
            y_windows.append(y_window)

        if len(x_windows) == 0:
            n_features = len(df.columns)
            return (np.zeros((0, input_chunk, n_features)), np.zeros((0, output_chunk)))

        return np.stack(x_windows), np.stack(y_windows)

    def _get_total_folds(self) -> int:
        """Get number of CV folds."""
        return getattr(self.splitter, "n_splits", -1)

    def initialize(
        self, approach: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any], int]:
        """
        Prepare RAW data for CV (no preprocessing leakage).

        Args:
            approach: Model configuration

        Returns:
            x_full_raw: RAW input windows (chưa normalize)
            y_full: Target windows
            model_name: Model name string
            hyperparams: Model hyperparameters
            total_folds: Number of folds
        """
        # 1. Load RAW data
        df_raw = self._load_raw_data()

        # 2. Create windows from RAW data (no normalization)
        input_chunk = approach.get("hyperparams", {}).get("input_chunk_length", 24)
        output_chunk = approach.get("hyperparams", {}).get("output_chunk_length", 6)

        x_full_raw, y_full = self._create_windows_raw(df_raw, input_chunk, output_chunk)

        # 3. Extract model info
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})
        total_folds = self._get_total_folds()

        print(f"[CVInitializer] Loaded RAW data: {x_full_raw.shape}")
        print(
            f"[CVInitializer] ⚠️  Data is NOT normalized - each fold will fit its own scaler"
        )

        return x_full_raw, y_full, model_name, hyperparams, total_folds
