from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from mlproject.src.datamodule.dataset import NumpyWindowDataset


class TSBaseDataModule:
    """Base class for ML/DL time-series DataModules.

    This module handles:
        - Sliding-window generation for supervised time-series.
        - Train/validation/test temporal splitting.
        - DataLoader instantiation.

    Attributes:
        df (pd.DataFrame): Input time-series dataframe.
        cfg (dict): Configuration dictionary.
        target_column (str): Name of the target variable.
        input_chunk (int): Length of each input window.
        output_chunk (int): Length of each output prediction horizon.
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray):
            Arrays generated after preprocessing.

    Raises:
        ValueError: If the dataframe is empty or windowing yields no samples.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        target_column: str,
        input_chunk: int,
        output_chunk: int,
    ) -> None:
        """Initialize the data module.

        Args:
            df (pd.DataFrame): Input dataframe containing features and target.
            cfg (dict): Configuration dictionary.
            target_column (str): Column name of the prediction target.
            input_chunk (int): Length of input sequence.
            output_chunk (int): Length of output sequence.

        Raises:
            ValueError: If the dataframe is empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("Input dataframe is empty.")

        self.df = df
        self.cfg = cfg
        self.target_column = target_column
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk

        self.x_train: np.ndarray
        self.y_train: np.ndarray
        self.x_val: np.ndarray
        self.y_val: np.ndarray
        self.x_test: np.ndarray
        self.y_test: np.ndarray

        self._prepare_data()

    def setup(self) -> None:
        """Optional hook to extend setup logic."""

    def _prepare_data(self) -> None:
        """Generate windows and perform temporal splits.

        Split ratios taken from:

        cfg["preprocessing"]["split"] = {"train": x, "val": y, "test": z}

        Raises:
            ValueError: If no windows can be created.
        """
        split_cfg = self.cfg.get("preprocessing", {}).get(
            "split",
            {"train": 0.6, "val": 0.2, "test": 0.2},
        )

        x, y = self._create_windows(
            target_col=self.target_column,
            input_chunk=self.input_chunk,
            output_chunk=self.output_chunk,
        )

        if len(x) == 0:
            raise ValueError("Windowing produced zero samples. Check input length.")

        n = len(x)
        n_train = int(n * split_cfg["train"])
        n_val = int(n * split_cfg["val"])

        self.x_train = x[:n_train]
        self.y_train = y[:n_train]

        self.x_val = x[n_train : n_train + n_val]
        self.y_val = y[n_train : n_train + n_val]

        self.x_test = x[n_train + n_val :]
        self.y_test = y[n_train + n_val :]

    def _create_windows(
        self,
        target_col: str,
        input_chunk: int,
        output_chunk: int,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the dataframe into sliding windows.

        Args:
            target_col (str): Name of the target variable.
            input_chunk (int): Input sequence length.
            output_chunk (int): Output horizon length.
            stride (int, optional): Window stride. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                x_windows: Array of shape (N, input_chunk, num_features)
                y_windows: Array of shape (N, output_chunk)

        Notes:
            A window is created only if both:
            - Input window is fully inside dataframe.
            - Output horizon fits within dataframe.
        """
        features = self.df.columns.tolist()
        n = len(self.df)

        x_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []

        for end_idx in range(input_chunk, n - output_chunk + 1, stride):
            start_idx = end_idx - input_chunk
            x_win = self.df.iloc[start_idx:end_idx].values
            y_win = self.df.iloc[end_idx : end_idx + output_chunk][target_col].values
            x_windows.append(x_win)
            y_windows.append(y_win)

        if not x_windows:
            return (
                np.zeros((0, input_chunk, len(features)), dtype=float),
                np.zeros((0, output_chunk), dtype=float),
            )

        return np.stack(x_windows), np.stack(y_windows)

    def get_data(self) -> Tuple[np.ndarray, ...]:
        """Get all train/val/test arrays.

        Returns:
            Tuple[np.ndarray, ...]: (x_train, y_train, x_val, y_val, x_test, y_test)

        Raises:
            AssertionError: If arrays are missing.
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.y_train, np.ndarray)
        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.y_val, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)
        assert isinstance(self.y_test, np.ndarray)

        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )

    def get_loaders(
        self, batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader, None, None]:
        """Create PyTorch DataLoaders for train/validation sets.

        Args:
            batch_size (int, optional): Loader batch size. Defaults to 16.

        Returns:
            Tuple[DataLoader, DataLoader, None, None]:
                (train_loader, val_loader, None, None)
        """
        train_loader = DataLoader(
            NumpyWindowDataset(self.x_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            NumpyWindowDataset(self.x_val, self.y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, None, None

    def summary(self) -> Tuple[int, int, int]:
        """Return summary of dataset split sizes.

        Returns:
            Tuple[int, int, int]: (num_train, num_val, num_test)

        Raises:
            AssertionError: If internal arrays are missing.
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.x_test, np.ndarray)

        return len(self.x_train), len(self.x_val), len(self.x_test)
