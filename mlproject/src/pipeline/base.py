from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf


class BasePipeline(ABC):
    """
    Abstract base pipeline defining the generic workflow.
    """

    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        cfg: DictConfig preferred, fallback empty DictConfig.
        """
        if cfg is None:
            self.cfg: DictConfig = DictConfig({})
        elif isinstance(cfg, dict):
            # convert dict → DictConfig
            self.cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, DictConfig):
            self.cfg = cfg
        else:
            raise TypeError("cfg must be dict or DictConfig")

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Run preprocessing and return a dataset."""

    @abstractmethod
    def run_approach(self, approach: Dict[str, Any], data: pd.DataFrame):
        """Run a single experiment approach."""

    def run(self, data: Optional[pd.DataFrame] = None):
        """
        Execute preprocessing and run all approaches.
        If data is provided, skip preprocessing.
        """
        if data is None:
            data = self.preprocess()

        # supports DictConfig for cfg
        approaches = OmegaConf.select(self.cfg, "experiment.approaches") or []

        if not approaches:
            raise RuntimeError("No approaches defined under experiment.approaches")

        for approach in approaches:
            print(f"\n=== Running approach: {approach.get('name', 'Unnamed')} ===")
            self.run_approach(approach, data)
