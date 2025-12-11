"""
Unit tests for CV splitters.

Run:
    pytest tests/test_cv_splitter.py -v
"""

import numpy as np
import pytest

from mlproject.src.datamodule.splitter import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)


class TestExpandingWindowSplitter:
    """Test cases for ExpandingWindowSplitter."""

    def test_basic_split(self):
        """Test basic expanding window split."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        x = np.arange(100).reshape(-1, 1, 1)
        y = np.arange(100).reshape(-1, 1)

        splits = list(splitter.split(x, y))

        # Should have 3 folds
        assert len(splits) == 3

        # Check fold 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 70  # 100 - 3*10 = 70
        assert len(test_idx) == 10
        assert test_idx[0] == 70
        assert test_idx[-1] == 79

        # Check fold 2
        train_idx, test_idx = splits[1]
        assert len(train_idx) == 80
        assert len(test_idx) == 10
        assert test_idx[0] == 80

        # Check fold 3
        train_idx, test_idx = splits[2]
        assert len(train_idx) == 90
        assert len(test_idx) == 10
        assert test_idx[0] == 90

    def test_no_overlap(self):
        """Test that train and test sets don't overlap."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        x = np.arange(100).reshape(-1, 1, 1)
        y = np.arange(100).reshape(-1, 1)

        for train_idx, test_idx in splitter.split(x, y):
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

            # Train always before test (time-series constraint)
            assert max(train_idx) < min(test_idx)

    def test_insufficient_data(self):
        """Test error when not enough data."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=20)

        # Only 50 samples, need 3*20 + 20 = 80
        x = np.arange(50).reshape(-1, 1, 1)
        y = np.arange(50).reshape(-1, 1)

        with pytest.raises(ValueError, match=r"(?i)insufficient data"):
            list(splitter.split(x, y))


class TestSlidingWindowSplitter:
    """Test cases for SlidingWindowSplitter."""

    def test_basic_split(self):
        """Test basic sliding window split."""
        splitter = SlidingWindowSplitter(n_splits=3, train_size=40, test_size=10)

        x = np.arange(100).reshape(-1, 1, 1)
        y = np.arange(100).reshape(-1, 1)

        splits = list(splitter.split(x, y))

        assert len(splits) == 3

        # Check fold 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 40
        assert len(test_idx) == 10

        # Check sliding property
        train_idx_1, _ = splits[0]
        train_idx_2, _ = splits[1]

        # Train window should slide forward
        assert min(train_idx_2) > min(train_idx_1)

    def test_fixed_train_size(self):
        """Test that train size stays constant."""
        splitter = SlidingWindowSplitter(n_splits=3, train_size=40, test_size=10)

        x = np.arange(100).reshape(-1, 1, 1)
        y = np.arange(100).reshape(-1, 1)

        for train_idx, _ in splitter.split(x, y):
            assert len(train_idx) == 40
