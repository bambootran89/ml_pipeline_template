"""
Feature retrieval strategies for different data types.

This module implements the Strategy pattern to handle timeseries
and tabular data loading from Feast with zero duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd

from mlproject.src.features.base import BaseFeatureStore
from mlproject.src.features.timeseries import TimeSeriesFeatureStore


class FeatureRetrievalStrategy(ABC):
    """
    Abstract strategy for feature retrieval from Feast.

    Implementations must define how to fetch features for a
    specific data type (timeseries, tabular, etc).
    """

    @abstractmethod
    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_id: Union[int, str],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Retrieve features from the feature store.

        Parameters
        ----------
        store : BaseFeatureStore
            Initialized feature store client.
        features : List[str]
            Fully qualified feature references.
        entity_key : str
            Entity join key name.
        entity_id : Union[int, str]
            Entity identifier value.
        config : Dict[str, Any]
            Additional configuration parameters.

        Returns
        -------
        pd.DataFrame
            Retrieved features.
        """


class TimeseriesRetrievalStrategy(FeatureRetrievalStrategy):
    """Strategy for timeseries feature retrieval with sequence windows."""

    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_id: Union[int, str],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Retrieve timeseries features as a sequence window."""
        ts_store = TimeSeriesFeatureStore(
            store=store,
            default_entity_key=entity_key,
            default_entity_id=entity_id,
        )

        # Extract time range
        start_date: datetime = datetime.fromisoformat(config["start_date"])
        end_date: datetime = datetime.fromisoformat(config["end_date"])

        return ts_store.get_sequence_by_range(
            features=features,
            start_date=start_date,
            end_date=end_date,
        )


class TabularRetrievalStrategy(FeatureRetrievalStrategy):
    """Strategy for tabular feature retrieval with entity-based joins."""

    def retrieve(
        self,
        store: BaseFeatureStore,
        features: List[str],
        entity_key: str,
        entity_id: Union[int, str],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Retrieve tabular features via historical or online lookup."""
        entity_data_path: str = config.get("entity_data", "")

        if not entity_data_path:
            raise ValueError("Tabular strategy requires 'entity_data' path")

        # Load entity reference data
        if entity_data_path.endswith(".csv"):
            entity_df = pd.read_csv(entity_data_path)
        elif entity_data_path.endswith(".parquet"):
            entity_df = pd.read_parquet(entity_data_path)
        else:
            raise ValueError(f"Unsupported format: {entity_data_path}")

        index_col: str = config.get("index_col", "event_timestamp")
        cols = [entity_key, index_col]
        entity_df = entity_df[cols]

        return store.get_historical_features(
            entity_df=entity_df,
            features=features,
        )


class StrategyFactory:
    """Factory for creating appropriate retrieval strategies."""

    _REGISTRY: Dict[str, type[FeatureRetrievalStrategy]] = {
        "timeseries": TimeseriesRetrievalStrategy,
        "tabular": TabularRetrievalStrategy,
    }

    @classmethod
    def create(cls, data_type: str) -> FeatureRetrievalStrategy:
        """
        Create a retrieval strategy for the given data type.

        Parameters
        ----------
        data_type : str
            Type of data ("timeseries" or "tabular").

        Returns
        -------
        FeatureRetrievalStrategy
            Strategy instance for the specified data type.

        Raises
        ------
        ValueError
            If data_type is not supported.
        """
        strategy_class = cls._REGISTRY.get(data_type.lower())

        if strategy_class is None:
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )

        return strategy_class()
