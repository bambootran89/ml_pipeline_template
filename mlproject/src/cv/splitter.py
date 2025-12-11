"""
Time-series cross-validation splitters.

Ensures chronological integrity and prevents data leakage
from future to past.
"""

from typing import Iterator, Tuple

import numpy as np


class TimeSeriesSplitter:
    """
    Base class for time-series cross-validation.

    Core principle:
        - Training data must come strictly from the past.
        - Test data must be strictly in the future relative to train.

    Subclasses must implement `split()`.
    """

    def split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate chronological train/test splits.

        Args:
            x: Array of input features with shape (N, seq_len, feat_dim).
            y: Array of target values with shape (N, horizon_dim).

        Yields:
            Tuple[np.ndarray, np.ndarray]: Train indices and test indices.

        Raises:
            NotImplementedError: If subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement the split() method.")


class ExpandingWindowSplitter(TimeSeriesSplitter):
    """
    Expanding-window time-series CV.

    Train window grows with each fold while test window remains fixed.

    Example:
        n = 100, n_splits = 3, test_size = 20

        Fold 1: train = [0:40],  test = [40:60]
        Fold 2: train = [0:60],  test = [60:80]
        Fold 3: train = [0:80],  test = [80:100]
    """

    def __init__(self, n_splits: int = 3, test_size: int = 20):
        """Initialize the splitter.

        Args:
            n_splits: Number of CV folds.
            test_size: Size of the test window for each fold.
        """
        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding-window splits.

        Args:
            x: Input feature array.
            y: Target array (unused, included for interface consistency).

        Yields:
            (train_indices, test_indices)

        Raises:
            ValueError: If dataset is too small for the requested split config.
        """
        n = len(x)
        total_test = self.n_splits * self.test_size
        min_train = n - total_test

        if min_train < self.test_size:
            raise ValueError(
                "Insufficient data for expanding-window CV. "
                f"Need ≥ {total_test + self.test_size}, got {n}."
            )

        for i in range(self.n_splits):
            test_start = min_train + i * self.test_size
            test_end = test_start + self.test_size
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


class SlidingWindowSplitter(TimeSeriesSplitter):
    """
    Sliding-window time-series CV.

    Train window size is fixed and slides forward with each fold.

    Example:
        n = 100, train_size = 40, test_size = 20, n_splits = 3

        Fold 1: train=[0:40],   test=[40:60]
        Fold 2: train=[20:60],  test=[60:80]
        Fold 3: train=[40:80],  test=[80:100]
    """

    def __init__(
        self,
        n_splits: int = 3,
        train_size: int = 40,
        test_size: int = 20,
    ):
        """Initialize sliding-window splitter.

        Args:
            n_splits: Number of folds.
            train_size: Fixed size of the train window.
            test_size: Size of the test window per fold.
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size

    def split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate sliding-window splits.

        Args:
            x: Feature array.
            y: Target array (unused).

        Yields:
            (train_indices, test_indices)

        Raises:
            ValueError: If train/test overlap or indices invalid.
        """
        n = len(x)

        for i in range(self.n_splits):
            remaining = (self.n_splits - i - 1) * self.test_size
            test_end = n - remaining
            test_start = test_end - self.test_size
            train_start = max(0, test_start - self.train_size)

            if train_start == test_start:
                raise ValueError(f"Invalid fold {i}: train and test overlap.")

            train_idx = np.arange(train_start, test_start)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
