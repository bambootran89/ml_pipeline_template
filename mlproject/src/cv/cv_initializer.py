"""
CVInitializer: Prepare cross-validation context.

Responsibilities:
- Build DataModule
- Extract full X/Y arrays
- Extract model name and hyperparameters
- Resolve number of folds
"""

from typing import Any, Dict, Tuple

import numpy as np
from omegaconf import DictConfig

from mlproject.src.cv.splitter import TimeSeriesSplitter
from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule


class CVInitializer:
    """Helper class to prepare CV context before fold execution."""

    def __init__(self, cfg: DictConfig, splitter: TimeSeriesSplitter) -> None:
        """
        Args:
            cfg: OmegaConf experiment configuration.
            splitter: TimeSeriesSplitter instance for CV splitting.
        """
        self.cfg = cfg
        self.splitter = splitter

    def _extract_full_dataset(self, dm: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract full X and Y arrays from supported DataModules.

        Args:
            dm: DataModule instance (TSDLDataModule or TSMLDataModule).

        Returns:
            x_full: Concatenated input array.
            y_full: Concatenated target array.
        """
        if isinstance(dm, (TSDLDataModule, TSMLDataModule)):
            x_full = np.concatenate([dm.x_train, dm.x_val, dm.x_test], axis=0)
            y_full = np.concatenate([dm.y_train, dm.y_val, dm.y_test], axis=0)
            return x_full, y_full
        raise TypeError(f"Unsupported DataModule type: {type(dm)}")

    def _get_total_folds(self) -> int:
        """
        Safely get the number of CV folds from the splitter.

        Returns:
            int: Number of folds, or -1 if unavailable.
        """
        return getattr(self.splitter, "n_splits", -1)

    def initialize(
        self, approach: Dict[str, Any], data: Any
    ) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any], int]:
        """
        Prepare all metadata needed for cross-validation.

        Args:
            approach: Dictionary with keys 'model' and optional 'hyperparams'.
            data: Raw input data for DataModule.

        Returns:
            Tuple containing:
                - x_full: np.ndarray of inputs
                - y_full: np.ndarray of targets
                - model_name: str
                - hyperparams: dict
                - total_folds: int
        """
        dm = DataModuleFactory.build(self.cfg, data)
        dm.setup()

        x_full, y_full = self._extract_full_dataset(dm)
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})
        total_folds = self._get_total_folds()

        return x_full, y_full, model_name, hyperparams, total_folds
