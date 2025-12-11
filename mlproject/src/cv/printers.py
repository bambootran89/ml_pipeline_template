"""
cv_printer.py

Utility module providing pretty-print helpers for cross-validation stages.
This centralizes all console output formatting used during the CV pipeline,
ensuring consistent and readable logs across folds and final summaries.
"""

from typing import Dict


class CVPrinter:
    """
    Pretty-print helper for cross-validation processes.

    This class contains only static utility methods, allowing the caller to
    print fold headers and aggregated CV results in a consistent format.

    Methods
    -------
    fold_header(fold_num, total_folds, train_size, test_size)
        Print a standardized header for a single CV fold.

    summary(aggregated)
        Print a formatted summary of aggregated CV metrics.
    """

    @staticmethod
    def fold_header(
        fold_num: int,
        total_folds: int,
        train_size: int,
        test_size: int,
    ) -> None:
        """
        Print a formatted header for a single cross-validation fold.

        Parameters
        ----------
        fold_num : int
            Index of the current CV fold (1-based).
        total_folds : int
            Total number of folds in the cross-validation process.
        train_size : int
            Number of samples used for training in this fold.
        test_size : int
            Number of samples used for testing in this fold.
        """
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_num}/{total_folds}")
        print(f"  Train: {train_size} samples, Test: {test_size} samples")
        print("=" * 60)

    @staticmethod
    def summary(aggregated: Dict[str, float]) -> None:
        """
        Print a formatted summary of aggregated CV metrics.

        Parameters
        ----------
        aggregated : Dict[str, float]
            A mapping of metric names to aggregated float values
            (typically mean metrics across folds).
        """
        print("\n" + "=" * 60)
        print("  CROSS-VALIDATION RESULTS")
        print("=" * 60)
        for key, value in aggregated.items():
            print(f"{key}: {value:.6f}")
        print("=" * 60 + "\n")
