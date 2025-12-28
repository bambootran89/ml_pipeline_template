"""
Specialized Feature Store wrapper for Time-Series Sequence operations.

This module provides a high-level API to retrieve sequences of data. It abstracts
away the requirement of entity IDs for single-series datasets by providing
internal default handling for entity keys and identifiers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

import pandas as pd

from mlproject.src.features.base import BaseFeatureStore


class TimeSeriesFeatureStore:
    """
    Wrapper for BaseFeatureStore to handle sequence-based retrieval.

    This class encapsulates the entity ID logic. For single-series datasets,
    the user does not need to provide an ID or entity key after initialization.
    """

    TIMESTAMP_FIELD = "event_timestamp"

    def __init__(
        self,
        store: BaseFeatureStore,
        default_entity_key: str = "location_id",
        default_entity_id: Union[int, str] = 1,
    ):
        """
        Initialize the wrapper with a store and default entity metadata.

        Args:
            store: An instance implementing the BaseFeatureStore interface.
            default_entity_key: Internal join key name (default: "location_id").
            default_entity_id: Internal identifier value (default: 1).
        """
        self.store = store
        self.default_entity_key = default_entity_key
        self.default_entity_id = default_entity_id

    def get_sequence_by_range(
        self,
        features: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1H",
        entity_id: Optional[Union[int, str]] = None,
        entity_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve a continuous sequence of features for a specific time range.

        Args:
            features: List of feature names (e.g., 'view:feature').
            start_date: Start of the time range (inclusive).
            end_date: End of the time range (inclusive).
            frequency: Pandas-style frequency string (default: '1H').
            entity_id: Optional override for the identifier.
            entity_key: Optional override for the join key.

        Returns:
            DataFrame containing the feature sequence sorted by time.
        """
        target_id = entity_id if entity_id is not None else self.default_entity_id
        target_key = entity_key if entity_key is not None else self.default_entity_key

        timestamps = pd.date_range(
            start=start_date, end=end_date, freq=frequency, tz=timezone.utc
        )

        entity_df = pd.DataFrame(
            {
                target_key: [target_id] * len(timestamps),
                self.TIMESTAMP_FIELD: timestamps,
            }
        )

        return self.store.get_historical_features(
            entity_df=entity_df, features=features
        ).sort_values(self.TIMESTAMP_FIELD)

    def get_latest_n_sequence(
        self,
        features: List[str],
        n_points: int,
        frequency_hours: int = 1,
        entity_id: Optional[Union[int, str]] = None,
        entity_key: Optional[str] = None,
        time_point: str = "now",
    ) -> pd.DataFrame:
        """
        Retrieve the N most recent feature points for inference.

        Args:
            features: List of feature names.
            n_points: Number of historical points to retrieve.
            frequency_hours: The hourly interval between points.
            entity_id: Optional override for the identifier.
            entity_key: Optional override for the join key.

        Returns:
            DataFrame of shape (n_points, len(features) + metadata).
        """
        target_id = entity_id if entity_id is not None else self.default_entity_id
        target_key = entity_key if entity_key is not None else self.default_entity_key
        if time_point == "now":
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            timestamps = [
                now - timedelta(hours=i * frequency_hours) for i in range(n_points)
            ]
        else:
            datetime_time_point = datetime.fromisoformat(time_point).astimezone(
                timezone.utc
            )
            timestamps = [
                datetime_time_point - timedelta(hours=i * frequency_hours)
                for i in range(n_points)
            ]

        entity_df = pd.DataFrame(
            {
                target_key: [target_id] * n_points,
                self.TIMESTAMP_FIELD: timestamps,
            }
        )

        return self.store.get_historical_features(
            entity_df=entity_df, features=features
        ).sort_values(self.TIMESTAMP_FIELD)
