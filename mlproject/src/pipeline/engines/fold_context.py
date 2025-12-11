from typing import Any

import numpy as np


class FoldContext:
    """
    Lightweight container holding all data and state needed to run
    a single cross-validation fold.

    It groups raw inputs, processed data, model objects, and evaluation
    outputs so `FoldRunner` can operate cleanly without many local variables.

    Parameters
    ----------
    fold_num : int
        Fold index.
    train_idx : Any
        Indices for training samples.
    test_idx : Any
        Indices for test/validation samples.
    x_train_raw : np.ndarray
        Raw training input features.
    y_train : np.ndarray
        Training target values.
    x_test_raw : np.ndarray
        Raw test input features.
    y_test : np.ndarray
        Test target values.
    is_tuning : bool
        Whether this fold is used for hyperparameter tuning.

    Attributes
    ----------
    preprocessor : Any
        Fitted preprocessing object for this fold.
    x_train_scaled : np.ndarray
        Scaled training data.
    x_test_scaled : np.ndarray
        Scaled test data.
    wrapper : Any
        Model wrapper instance.
    trainer : Any
        Trainer instance.
    metrics : dict | None
        Evaluation metrics after inference.
    """

    def __init__(
        self,
        fold_num: int,
        train_idx: Any,
        test_idx: Any,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_test_raw: np.ndarray,
        y_test: np.ndarray,
        is_tuning: bool,
    ):
        self.fold_num = fold_num
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.x_train_raw = x_train_raw
        self.y_train = y_train
        self.x_test_raw = x_test_raw
        self.y_test = y_test
        self.is_tuning = is_tuning

        # Filled progressively during the fold pipeline
        self.preprocessor = None
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.wrapper = None
        self.trainer = None
        self.metrics = None
