from typing import Tuple

import numpy as np

from mlproject.src.datamodule.tsbase import TSBaseDataModule


class TSMLDataModule(TSBaseDataModule):
    """
    ML DataModule for XGBoost / sklearn / traditional ML.
    Returns numpy arrays (train/val/test) with optional feature selection.
    """

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
