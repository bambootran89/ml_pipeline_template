from typing import Optional, Tuple

import numpy as np
import pandas as pd


class TSBaseDataModule:
    """
    Base class for ML / DL DataModules.
    Handles:
    - time-series windowing
    - train/val/test splitting
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        target_column: str,
        input_chunk: int,
        output_chunk: int,
    ):
        self.df = df
        self.cfg = cfg
        self.target_column = target_column
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk

        # will be filled by _prepare_data()
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self._prepare_data()

    def _prepare_data(self):
        """Windowing + temporal split."""
        sp = self.cfg.get("preprocessing", {}).get(
            "split", {"train": 0.6, "val": 0.2, "test": 0.2}
        )

        x, y = self._create_windows(
            self.target_column,
            self.input_chunk,
            self.output_chunk,
        )

        n = len(x)
        n_train = int(n * sp["train"])
        n_val = int(n * sp["val"])

        self.x_train = x[:n_train]
        self.y_train = y[:n_train]

        self.x_val = x[n_train : n_train + n_val]
        self.y_val = y[n_train : n_train + n_val]

        self.x_test = x[n_train + n_val :]
        self.y_test = y[n_train + n_val :]

    def _create_windows(self, target_col, input_chunk, output_chunk, stride=1):
        """
        Convert a DataFrame into input/output windows for training.

        Args:
            df (pd.DataFrame): Input dataframe with features and target.
            target_col (str): Target column name.
            input_chunk (int): Length of input sequence.
            output_chunk (int): Length of output sequence.
            stride (int): Sliding window step.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            x_windows (N, seq_len, feat), y_windows (N, horizon)
        """
        features = self.df.columns.tolist()
        n = len(self.df)
        x_windows, y_windows = [], []

        for end_idx in range(input_chunk, n - output_chunk + 1, stride):
            start_idx = end_idx - input_chunk
            x_window = self.df.iloc[start_idx:end_idx].values  # (seq_len, feat)
            y_window = self.df.iloc[end_idx : end_idx + output_chunk][
                target_col
            ].values  # (output_chunk,)
            x_windows.append(x_window)
            y_windows.append(y_window)

        if len(x_windows) == 0:
            return np.zeros((0, input_chunk, len(features))), np.zeros(
                (0, output_chunk)
            )

        return np.stack(x_windows), np.stack(y_windows)

    def summary(self) -> Tuple[int, int, int]:
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)
        return len(self.x_train), len(self.x_val), len(self.x_test)
