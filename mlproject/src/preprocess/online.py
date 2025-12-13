from typing import Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .engine import PreprocessEngine


class OnlinePreprocessor:
    """
    Wrapper class for Online Preprocessing using PreprocessEngine.
    Designed to be instantiated ONCE during service startup.
    """

    def __init__(self, cfg: Optional[DictConfig] = None):
        """
        Initialize the preprocessor and underlying engine once.

        Args:
            cfg (DictConfig, optional): Preprocessing configuration.
        """
        self.engine = PreprocessEngine(is_train=False, cfg=self._to_dict(cfg))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the dataframe using the persistent engine.

        Args:
            df (pd.DataFrame): Raw input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe ready for inference.
        """
        # Gọi hàm online_transform của engine đã khởi tạo
        return self.engine.online_transform(df)

    def update_config(self, cfg: DictConfig):
        """
        Bridge to update the underlying engine's configuration.
        """
        cfg_dict = self._to_dict(cfg)
        self.engine.update_config(cfg_dict)

    def _to_dict(self, cfg: Optional[DictConfig]) -> dict:
        """Helper to convert OmegaConf to dict safely."""
        if cfg is None:
            return {}
        container = OmegaConf.to_container(cfg, resolve=True)
        return container if isinstance(container, dict) else {}
