"""
Feast Feature Store dataset loader for offline historical time-series features.

Supports:
- 'timeseries': Fetch historical sequences using TimeSeriesFeatureStore.
- 'tabular': Fetch latest or historical features for entities.
"""

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
    """Dataset loader retrieving historical
    time-series and tabular features from Feast."""

    TIMESTAMP_FIELD: str = "event_timestamp"
    SUPPORTED_TYPE: list[str] = ["timeseries", "tabular"]
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
        """
        Load historical features from a Feast offline feature repository.

        Args:
            cfg: DictConfig object with feature metadata.
            path: Feast URI path (feast://<repo>?entity=<key>&id=<val>).
            index_col: Optional column to set as DataFrame index.
            data_type: 'timeseries' or 'tabular'.

        Returns:
            DataFrame with requested features.

        Raises:
            ValueError: If required configs are missing or data_type unsupported.
        """
        self._validate_data_type(data_type)
        repo_path, entity_key, entity_id, _ = self._parse_uri(path)

        self._validate_config_attrs(cfg, ["features", "featureview"])
        full_features = [f"{cfg.data.featureview}:{f}" for f in cfg.data.features]
        full_target_columns = [
            f"{cfg.data.featureview}:{f}" for f in cfg.data.get("target_columns", [])
        ]
        if data_type == "tabular":
            return self._load_tabular(
                cfg, repo_path, full_features + full_target_columns
            )

        return self._load_timeseries(
            cfg, repo_path, entity_key, entity_id, full_features, index_col
        )

    def _validate_data_type(self, data_type: str) -> None:
        """Validate data_type input."""
        if data_type not in self.SUPPORTED_TYPE:
            raise ValueError(
                f"FeastDatasetLoader only supports types: {self.SUPPORTED_TYPE}"
            )

    def _validate_config_attrs(self, cfg: DictConfig, attrs: list[str]) -> None:
        """Ensure required attributes exist in cfg.data."""
        for attr in attrs:
            if not cfg or not hasattr(cfg.data, attr):
                raise ValueError(f"Missing '{attr}' in config YAML (cfg.data)")

    def _load_tabular(
        self, cfg: DictConfig, repo_path: str, full_features: list[str]
    ) -> pd.DataFrame:
        """
        Load tabular features from Feast (offline or online store).

        Steps:
        1. Initialize Feast store via Factory.
        2. Load entity data from CSV or Parquet as specified in cfg.
        3. Fetch historical features for these entities.
        4. Return resulting DataFrame with requested features.
        """
        self._validate_config_attrs(cfg, ["entity_data", "index_col", "entity_key"])

        base_store = FeatureStoreFactory.create(store_type="feast", repo_path=repo_path)

        cols: list[str] = [cfg.data.entity_key, cfg.data.index_col]
        entity_data_path: str = cfg.data.entity_data

        if entity_data_path.endswith(".csv"):
            entity_df: pd.DataFrame = pd.read_csv(entity_data_path)[cols]
        elif entity_data_path.endswith(".parquet"):
            entity_df = pd.read_parquet(entity_data_path)[cols]
        else:
            raise ValueError(f"Unsupported entity_data format: {entity_data_path}")

        df = base_store.get_historical_features(
            entity_df=entity_df, features=full_features
        )
        df = df.sort_values("passenger_id")
        print("\n[Feast Profiling] DataFrame result:")
        print(f"  → rows    : {len(df)}")
        print(f"  → columns : {list(df.columns)}")
        print("\n[Feast Profiling] head Sample:")
        print(df.head(5))
        print("\n[Feast Profiling] tail Sample:")
        print(df.tail(5))

        return df

    def _load_timeseries(
        self,
        cfg: DictConfig,
        repo_path: str,
        entity_key: str,
        entity_id: Union[int, str],
        full_features: list[str],
        index_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Load historical timeseries features using TimeSeriesFeatureStore.

        Steps:
        1. Initialize TimeSeriesFeatureStore.
        2. Parse start/end dates from config.
        3. Fetch sequence from Feast within date range.
        4. Optionally set index and sort by timestamp.
        """
        self._validate_config_attrs(cfg, ["start_date", "end_date"])

        ts_store = TimeSeriesFeatureStore(
            FeatureStoreFactory.create(store_type="feast", repo_path=repo_path),
            default_entity_key=entity_key,
            default_entity_id=entity_id,
        )

        cfg_start = datetime.fromisoformat(cfg.data.start_date).astimezone(timezone.utc)
        cfg_end = datetime.fromisoformat(cfg.data.end_date).astimezone(timezone.utc)

        print("[Feast Profiling] Requested range:")
        print(f"  → start: {cfg_start.isoformat()}")
        print(f"  → end  : {cfg_end.isoformat()}")
        print(f"  → total hours: {(cfg_end - cfg_start).total_seconds()/3600:.2f}")

        try:
            df = ts_store.get_sequence_by_range(
                features=full_features,
                start_date=cfg_start,
                end_date=cfg_end,
            )
            print("\n[Feast Profiling] DataFrame result:")
            print(f"  → rows    : {len(df)}")
            print(f"  → columns : {list(df.columns)}")
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

        Returns:
            (repo_path, entity_key, entity_id, empty_string)
        """
        if not path.startswith(self.URI_PREFIX):
            raise ValueError(f"Feast path must start with '{self.URI_PREFIX}'")

        body = path[len(self.URI_PREFIX) :]
        repo_path, _, query = body.partition("?")

        params: dict[str, str] = {}
        for pair in query.split("&"):
            if "=" in pair:
                k, _, v = pair.partition("=")
                params[k] = v

        entity_key = params.get("entity", self.default_entity_key)
        raw_id = params.get("id", str(self.default_entity_id))
        entity_id: Union[int, str] = int(raw_id) if raw_id.isdigit() else raw_id

        return repo_path, entity_key, entity_id, ""
