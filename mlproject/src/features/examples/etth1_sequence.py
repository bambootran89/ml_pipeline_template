"""
ETTh1 sequence retrieval demo with encapsulated entity IDs.

This script demonstrates how to preprocess ETTh1 data and retrieve
time-based feature sequences from Feast using a clean, high-level API
without repeatedly specifying entity IDs in each call.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager
from mlproject.src.features.timeseries import TimeSeriesFeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_etth1_data(raw_path: Path, processed_path: Path) -> None:
    """Preprocess ETTh1 data and shift timestamps to the current UTC hour."""
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])

    now = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    last_record_time = df["date"].iloc[-1].tz_localize(timezone.utc)
    time_offset = now - last_record_time

    df["event_timestamp"] = (
        df["date"].dt.tz_localize(
            timezone.utc,
        )
        + time_offset
    )

    df["location_id"] = 1

    df["hour"] = df["event_timestamp"].dt.hour
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24,
    ).astype(np.float32)

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24,
    ).astype(np.float32)

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(processed_path)
    logger.info(
        "ETTh1 data prepared and saved to %s",
        processed_path,
    )


def run_etth1_demo() -> None:
    """Run sequence retrieval for ETTh1 using a time series feature wrapper."""
    repo_name = "etth1_repo"
    raw_csv = Path("mlproject/data/ETTh1.csv")
    repo_path = Path(repo_name)
    processed_parquet = repo_path / "data" / "etth1_ready.parquet"

    FeastRepositoryManager.initialize_repo(repo_name)
    prepare_etth1_data(raw_csv, processed_parquet)

    base_store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    base_store.register_entity(
        "location",
        "location_id",
        value_type="int",
    )

    schema: dict[str, str] = {
        "HUFL": "float",
        "MUFL": "float",
        "mobility_inflow": "float",
        "hour_sin": "float",
        "hour_cos": "float",
    }

    base_store.register_feature_view(
        name="etth1_view",
        entities=["location"],
        schema=schema,
        source_path=str(processed_parquet.absolute()),
        ttl_days=365,
    )

    ts_store = TimeSeriesFeatureStore(
        base_store,
        default_entity_key="location_id",
        default_entity_id=1,
    )

    now = datetime.now(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    start = now - timedelta(hours=24)

    logger.info("Fetching 24h feature range for ETTh1 training...")
    df_train = ts_store.get_sequence_by_range(
        features=["etth1_view:HUFL", "etth1_view:mobility_inflow"],
        start_date=start,
        end_date=now,
    )

    print("\n--- ETTh1 Sequence Data (Training Range) ---")
    print(f"Total rows: {len(df_train)}")
    print(df_train.head())

    logger.info("Fetching latest 10 feature points for ETTh1 inference...")
    df_inf = ts_store.get_latest_n_sequence(
        features=["etth1_view:HUFL", "etth1_view:hour_sin"],
        n_points=10,
    )

    print("\n--- ETTh1 Latest N Sequence (Inference) ---")
    print(f"Total rows: {len(df_inf)}")
    print(df_inf)


if __name__ == "__main__":
    run_etth1_demo()
