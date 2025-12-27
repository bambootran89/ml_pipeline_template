"""
Generic station time-series sequence retrieval demo.

This script demonstrates preprocessing of station production data and
retrieval of time-based feature sequences from Feast using a simplified API
where the entity key 'station_id' is encapsulated in the
TimeSeriesFeatureStore wrapper to avoid repetitive passing in calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.timeseries import TimeSeriesFeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_environment(repo_name: str) -> Path:
    """Initialize a Feast repository and generate synthetic station data."""
    FeastRepositoryManager.initialize_repo(repo_name)
    repo_path = Path(repo_name)
    data_dir = repo_path / "data"
    data_dir.mkdir(
        exist_ok=True,
    )

    periods = 200
    now = datetime.now(timezone.utc)

    df = pd.DataFrame(
        {
            "station_id": [101] * periods,
            "event_timestamp": pd.date_range(
                end=now,
                periods=periods,
                freq="H",
                tz=timezone.utc,
            ),
            "production": np.random.uniform(
                50,
                100,
                size=periods,
            ).astype(np.float32),
        }
    )

    data_path = data_dir / "production.parquet"
    df.to_parquet(data_path)
    return data_path


def run_ts_workflow() -> None:
    """Execute sequence retrieval using a time series feature store wrapper."""
    repo_name = "ts_gen_repo"
    data_path = prepare_environment(repo_name)

    base_store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    base_store.register_entity(
        name="station",
        join_key="station_id",
        description="Production station",
        value_type="int",
    )

    station_entity = Entity(
        name="station",
        value_type=ValueType.INT64,
    )

    source = FileSource(
        path=str(data_path.absolute()),
        timestamp_field="event_timestamp",
    )

    view = FeatureView(
        name="station_stats",
        entities=[station_entity],
        schema=[Field(name="production", dtype=Float32)],
        source=source,
        ttl=timedelta(days=1),
    )

    if hasattr(base_store, "store") and view.source is not None:
        base_store.store.apply([view])

    ts_store = TimeSeriesFeatureStore(
        base_store,
        default_entity_key="station_id",
        default_entity_id=101,
    )

    logger.info("Fetching sequence range for model training...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=10)

    df_range = ts_store.get_sequence_by_range(
        features=["station_stats:production"],
        start_date=start,
        end_date=end,
    )

    print("\n--- Station Range Retrieval Result ---")
    print(f"Total rows: {len(df_range)}")
    print(df_range.head())

    logger.info("Fetching latest 5 feature points for inference...")
    df_latest = ts_store.get_latest_n_sequence(
        features=["station_stats:production"],
        n_points=5,
    )

    print("\n--- Station Latest N Retrieval Result ---")
    print(f"Total rows: {len(df_latest)}")
    print(df_latest)


if __name__ == "__main__":
    run_ts_workflow()
