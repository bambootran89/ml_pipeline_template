from typing import Tuple

import numpy as np
import pandas as pd

from mlproject.src.preprocess.base import PreprocessBase


class FoldPreprocessor:
    """
    Fold-specific preprocessing helper.

    This class fits a fresh preprocessor (e.g., scaler) on the training
    data of a single fold and applies the same transformation to both
    train and validation/test sets. It ensures each fold is fully
    isolated and avoids information leakage.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_names = cfg.get("data", {}).get("features", [])
        if not self.feature_names:
            raise ValueError("Feature names must be provided in config.")

    def _feature_names(self):
        """Return the feature names expected by the preprocessor."""
        return self.feature_names

    def fit(self, x_train: np.ndarray) -> Tuple[PreprocessBase, np.ndarray]:
        """
        Fit the preprocessor on raw training data and return the scaled output.

        Parameters
        ----------
        x_train : np.ndarray
            Raw training input of shape (n_samples, seq_len, n_features).

        Returns
        -------
        tuple
            (preprocessor, x_scaled), where:
                - preprocessor: fitted PreprocessBase instance
                - x_scaled: scaled training array reshaped to original dimensions
        """
        preprocessor = PreprocessBase(self.cfg)

        n_samples, seq_len, n_feat = x_train.shape
        df = pd.DataFrame(x_train.reshape(-1, n_feat), columns=self._feature_names())
        df_scaled = preprocessor.fit(df)
        x_scaled = df_scaled.values.reshape(n_samples, seq_len, n_feat)
        return preprocessor, x_scaled

    def transform(self, x: np.ndarray, preprocessor: PreprocessBase) -> np.ndarray:
        """
        Apply a fitted preprocessor to new data (e.g., validation/test).

        Parameters
        ----------
        x : np.ndarray
            Raw input data to be transformed.
        preprocessor : PreprocessBase
            The preprocessor previously fitted on training data.

        Returns
        -------
        np.ndarray
            Transformed data with the same shape as the input.
        """
        n_samples, seq_len, n_feat = x.shape
        df = pd.DataFrame(x.reshape(-1, n_feat), columns=self._feature_names())

        df_scaled = preprocessor.transform(df)
        return df_scaled.values.reshape(n_samples, seq_len, n_feat)
