from typing import Optional, Tuple

import numpy as np
import pandas as pd

from mlproject.src.data.base_datamodule import BaseDataModule
from mlproject.src.data.dataloader import create_windows


class MLDataModule(BaseDataModule):
    """
    ML DataModule for XGBoost / sklearn / traditional ML.
    Returns numpy arrays (train/val/test) with optional feature selection.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: dict,
        target_column: str,
    ):
        """
        Args:
            df: full DataFrame
            cfg: configuration dictionary
            target_column: name of target column
            feature_cols: optional list of feature columns
        """
        super().__init__(df, cfg, target_column)

        # Time-series chunks (optional, for consistency)
        self.input_chunk: Optional[int] = None
        self.output_chunk: Optional[int] = None

        # Arrays
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def setup(self, input_chunk: int, output_chunk: int):
        """Prepare windowed arrays from train/val/test DataFrames."""
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk

        # Add feature columns + target for create_windows

        self.x_train, self.y_train = create_windows(
            self.train_df, self.target_column, input_chunk, output_chunk
        )
        self.x_val, self.y_val = create_windows(
            self.val_df, self.target_column, input_chunk, output_chunk
        )
        self.x_test, self.y_test = create_windows(
            self.test_df, self.target_column, input_chunk, output_chunk
        )

    def get_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all arrays in order: train, val, test."""
        assert self.x_train is not None
        assert self.y_train is not None
        assert self.x_val is not None
        assert self.y_val is not None
        assert self.x_test is not None
        assert self.y_test is not None

        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )
