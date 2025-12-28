"""
Unified facade for feature store operations.

This module provides a simplified, high-level interface for
loading features from Feast regardless of data type.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.strategies import StrategyFactory


class FeatureStoreFacade:
    """
    Simplified facade for Feast feature retrieval.

    This class eliminates complexity by routing data type-specific
    logic to appropriate strategies while maintaining a clean API.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize facade with configuration.

        Parameters
        ----------
        cfg : DictConfig
            Configuration containing data and feature metadata.
        """
        self.cfg = cfg
        self.data_cfg = cfg.get("data", {})

    def load_features(self) -> pd.DataFrame:
        """
        Load features from Feast using configuration metadata.

        Returns
        -------
        pd.DataFrame
            Retrieved features ready for preprocessing.

        Raises
        ------
        ValueError
            If configuration is incomplete or invalid.
        """
        # Parse Feast URI
        uri: str = self.data_cfg.get("path", "")
        parsed = urlparse(uri)

        if not parsed.scheme or parsed.scheme != "feast":
            raise ValueError(f"Invalid Feast URI: {uri}")

        repo_name: str = parsed.netloc
        data_type: str = self.data_cfg.get("type", "timeseries")

        # Validate required config
        self._validate_config()

        # Initialize store
        store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=repo_name,
        )

        # Build feature references
        featureview: str = self.data_cfg["featureview"]
        only_features: List[str] = [
            f"{featureview}:{f}" for f in self.data_cfg["features"]
        ]
        target_columns: List[str] = [
            f"{featureview}:{f}" for f in self.data_cfg.get("target_columns", [])
        ]
        features = list(set(only_features + target_columns))
        # Extract entity metadata
        entity_key, entity_id = self._resolve_entity()

        # Prepare strategy config
        strategy_config = self._build_strategy_config()

        # Load using appropriate strategy
        strategy = StrategyFactory.create(data_type)
        df = strategy.retrieve(
            store=store,
            features=features,
            entity_key=entity_key,
            entity_id=entity_id,
            config=strategy_config,
        )

        return self._post_process(df)

    def _validate_config(self) -> None:
        """Validate required configuration keys."""
        required = ["featureview", "features"]
        missing = [k for k in required if k not in self.data_cfg]

        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _resolve_entity(self) -> tuple[str, Union[int, str]]:
        """
        Resolve entity key and ID from configuration.

        Returns
        -------
        tuple[str, Union[int, str]]
            Entity key name and entity ID value.
        """
        entity_key: str = self.data_cfg.get("entity_key", "location_id")
        entity_id: Union[int, str] = self.data_cfg.get("entity_id", 1)
        return entity_key, entity_id

    def _build_strategy_config(self) -> Dict[str, Any]:
        """
        Build configuration dict for strategy execution.

        Returns
        -------
        Dict[str, Any]
            Strategy-specific configuration.
        """
        return {
            "start_date": self.data_cfg.get("start_date", ""),
            "end_date": self.data_cfg.get("end_date", ""),
            "entity_data": self.data_cfg.get("entity_data", ""),
            "index_col": self.data_cfg.get("index_col", "event_timestamp"),
        }

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to retrieved features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw features from Feast.

        Returns
        -------
        pd.DataFrame
            Processed features ready for modeling.
        """
        index_col = self.data_cfg.get("index_col", "event_timestamp")

        if index_col in df.columns:
            df = df.set_index(index_col)

        feature_cols = self.data_cfg.get("features", [])
        target_cols = self.data_cfg.get("target_columns", [])
        available = [c for c in set(feature_cols + target_cols) if c in df.columns]

        if available:
            return df[available]

        return df
