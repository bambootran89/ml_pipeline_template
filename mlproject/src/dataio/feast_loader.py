"""Feast Feature Store dataset loader for offline historical features."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

import pandas as pd

from mlproject.src.dataio.base import BaseDatasetLoader
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.timeseries import TimeSeriesFeatureStore

logger = logging.getLogger(__name__)


class FeastDatasetLoader(BaseDatasetLoader):
    """Dataset loader retrieving historical time-series features from Feast."""

    TIMESTAMP_FIELD: str = "event_timestamp"
    SUPPORTED_TYPE: str = "timeseries"
    LOOKBACK_DAYS: int = 90
    URI_PREFIX: str = "feast://"

    default_entity_key: str = "location_id"
    default_entity_id: Union[int, str] = 1

    def load(
        self,
        path: str,
        *,
        index_col: Optional[str] = None,
        data_type: str,
    ) -> pd.DataFrame:
        """Load historical features from a Feast offline feature repository.

        Args:
            path: Feast repository path with optional query parameters.
            index_col: Optional timestamp column to set as index.
            data_type: Must be "timeseries" for this loader.

        Returns:
            DataFrame containing historical features sorted by timestamp.

        Raises:
            ValueError: If data_type is not supported or URI format is invalid.
            OSError: If the underlying feature store cannot be initialized.
        """
        if data_type != self.SUPPORTED_TYPE:
            raise ValueError(
                f"FeastDatasetLoader only supports data_type='{self.SUPPORTED_TYPE}'"
            )

        repo_path, entity_key, entity_id, features = self._parse_uri(path)

        base_store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=repo_path,
        )

        ts_store = TimeSeriesFeatureStore(
            base_store,
            default_entity_key=entity_key,
            default_entity_id=entity_id,
        )

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.LOOKBACK_DAYS)

        df = ts_store.get_sequence_by_range(
            features=features.split(","),
            start_date=start_date,
            end_date=end_date,
        )

        if index_col and index_col in df.columns:
            df = df.set_index(index_col)

        df = df.sort_values(self.TIMESTAMP_FIELD)

        logger.info(
            "Loaded %d rows from Feast (range: %s → %s)",
            len(df),
            start_date.date().isoformat(),
            end_date.date().isoformat(),
        )

        return df

    def _parse_uri(self, path: str) -> Tuple[str, str, Union[int, str], str]:
        """Parse Feast repository URI.

        Supported format:
            feast://<repo_path>?entity=<key>&id=<val>&features=<list>

        Args:
            path: URI string to parse.

        Returns:
            Tuple of (repo_path, entity_key, entity_id, features).

        Raises:
            ValueError: If URI prefix or parameters are invalid.
        """
        if not path.startswith(self.URI_PREFIX):
            raise ValueError(f"Feast path must start with '{self.URI_PREFIX}'")

        body = path[len(self.URI_PREFIX) :]
        repo_path, _, query = body.partition("?")

        params: dict[str, str] = {}
        for pair in query.split("&"):
            if not pair or "=" not in pair:
                continue
            key, _, val = pair.partition("=")
            params[key] = val

        entity_key = params.get("entity", self.default_entity_key)
        raw_id = params.get("id", str(self.default_entity_id))
        entity_id: Union[int, str] = int(raw_id) if raw_id.isdigit() else raw_id
        features = params.get("features", "")

        return repo_path, entity_key, entity_id, features
