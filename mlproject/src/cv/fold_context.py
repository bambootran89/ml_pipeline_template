"""
FoldContext: Holds all slice-level information for one CV fold.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FoldContext:
    """Container for per-fold information."""

    fold_num: int
    total_folds: int

    train_idx: Any
    test_idx: Any

    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any

    model_name: str
    hyperparams: dict
