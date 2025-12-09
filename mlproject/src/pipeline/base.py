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
        Execute preprocessing and run the experiment.
        If data is provided, skip preprocessing.
        """
        if data is None:
            data = self.preprocess()

        # Lấy experiment config trực tiếp (không còn approaches list)
        experiment = OmegaConf.select(self.cfg, "experiment")

        if not experiment:
            raise RuntimeError("No experiment config found")

        print(f"\n=== Running experiment: {experiment.get('name', 'Unnamed')} ===")
        self.run_approach(experiment, data)
