"""Feast Feature Store dataset loader for offline historical time-series features."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Union

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.dataio.base import BaseDatasetLoader
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.timeseries import TimeSeriesFeatureStore

logger = logging.getLogger(__name__)


class FeastDatasetLoader(BaseDatasetLoader):
    """Dataset loader retrieving historical time-series features from Feast."""

    TIMESTAMP_FIELD: str = "event_timestamp"
    SUPPORTED_TYPE: str = "timeseries"
    URI_PREFIX: str = "feast://"

    default_entity_key: str = "location_id"
    default_entity_id: Union[int, str] = 1

    def load(
        self,
        cfg: DictConfig,
        path: str,
        *,
        index_col: Optional[str] = None,
        data_type: str,
    ) -> pd.DataFrame:
        """Load historical features from a Feast offline feature repository."""

        if data_type != self.SUPPORTED_TYPE:
            raise ValueError(
                f"FeastDatasetLoader only supports data_type='{self.SUPPORTED_TYPE}'"
            )

        repo_path, entity_key, entity_id, _ = self._parse_uri(path)

        if not cfg or not hasattr(cfg.data, "features"):
            raise ValueError("Missing Feast feature list in config YAML.")

        if not cfg or not hasattr(cfg.data, "featureview"):
            raise ValueError("Missing Feast featureview in config YAML.")

        if not cfg or not hasattr(cfg.data, "start_date"):
            raise ValueError("Missing Feast start_date in config YAML.")

        if not cfg or not hasattr(cfg.data, "end_date"):
            raise ValueError("Missing Feast end_date in config YAML.")

        full_features = [f"{cfg.data.featureview}:{f}" for f in cfg.data.features]

        ts_store = TimeSeriesFeatureStore(
            FeatureStoreFactory.create(
                store_type="feast",
                repo_path=repo_path,
            ),
            default_entity_key=entity_key,
            default_entity_id=entity_id,
        )

        # Apply minimal time-range patch to avoid future timestamp filling
        cfg_start = datetime.fromisoformat(cfg.data.start_date).astimezone(timezone.utc)
        cfg_end = datetime.fromisoformat(cfg.data.end_date).astimezone(timezone.utc)

        print("[Feast Profiling] Requested range:")
        print(f"  → start: {cfg_start.isoformat()}")
        print(f"  → end  : {cfg_end.isoformat()}")
        print(
            f"  → total hours: " f"{(cfg_end - cfg_start).total_seconds() / 3600:.2f}"
        )

        try:
            df = ts_store.get_sequence_by_range(
                features=full_features,
                start_date=cfg_start,
                end_date=cfg_end,
            )
            print("\n[Feast Profiling] DataFrame result:")
            print(f"  → rows         : {len(df)}")
            print(f"  → columns      : {list(df.columns)}")

            print("\n[Feast Profiling] head Sample:")
            print(df.head(5))

            print("\n[Feast Profiling] tail Sample:")
            print(df.tail(5))

        except Exception as exc:
            print(f"[Feast Profiling] ERROR: {exc}")
            raise

        if df.empty:
            raise ValueError(
                f"No valid data from Feast repository '{repo_path}'. Ensure history "
                "exists and populate_feast.py was executed before loading."
            )

        if index_col and index_col in df.columns:
            df = df.set_index(index_col)

        df = df.sort_values(self.TIMESTAMP_FIELD)

        logger.info("Loaded %d rows from Feast.", len(df))
        return df

    def _parse_uri(self, path: str) -> Tuple[str, str, Union[int, str], str]:
        """Parse a Feast repository URI.

        Format:
            feast://<repo_path>?entity=<entity_key>&id=<entity_id>

        The features are intentionally ignored because they are read from config.

        Returns:
            (repo_path, entity_key, entity_id, empty_feature_string)
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

        return repo_path, entity_key, entity_id, ""
