"""
Time-series forecasting workflow demo using Feast Feature Store.

This module simulates hourly sensor telemetry, registers entities and
feature views, retrieves offline historical features with point-in-time
semantics, materializes features into the online store, and fetches
latest feature values for real-time inference using UTC timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager


def generate_ts_data(repo: Path, periods: int = 200) -> Path:
    """Generate synthetic UTC-aware sensor data with lag and rolling features."""
    file_path = repo / "data" / "ts_features.parquet"
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=periods),
        periods=periods,
        freq="H",
        tz=timezone.utc,
    )

    df = pd.DataFrame(
        {
            "location_id": [1] * periods,
            "event_timestamp": ts,
            "temperature": np.random.randn(periods).astype(float) * 10 + 25,
            "demand": np.random.randn(periods).astype(float) * 100 + 500,
        }
    )

    df["temp_lag_24"] = df["temperature"].shift(24)
    df["demand_roll_12"] = df["demand"].rolling(12).mean()

    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """Initialize Feast store and register forecasting entity and feature view."""
    store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    print("Registering entity 'location' for forecasting joins...")
    store.register_entity(
        name="location",
        join_key="location_id",
        description="Sensor location entity for time-series joins",
        value_type="int",
    )

    print("Registering feature view 'forecast_view'...")
    store.register_feature_view(
        name="forecast_view",
        entities=["location"],
        schema={
            "temp_lag_24": "float",
            "demand_roll_12": "float",
        },
        source_path=str(data_file),
        ttl_days=7,
    )

    return store


def main() -> None:
    """Run the complete time-series forecasting feature store workflow."""
    repo_name = "ts_repo"
    repo = Path(repo_name)

    FeastRepositoryManager.initialize_repo(repo_name)
    data_file = generate_ts_data(repo)

    store = build_store(repo_name, data_file)

    print("\n--- Offline Point-in-Time Historical Feature Retrieval ---")
    now = datetime.now(timezone.utc)
    entity_df: pd.DataFrame = pd.DataFrame(
        {
            "location_id": [1, 1, 1],
            "event_timestamp": [
                now - timedelta(hours=1),
                now - timedelta(hours=25),
                now - timedelta(hours=50),
            ],
        }
    )

    hist_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "forecast_view:temp_lag_24",
            "forecast_view:demand_roll_12",
        ],
    )

    print("Historical feature sample (first 5 rows):")
    print(hist_df.head())

    print("\n--- Materializing Features to Online Store ---")
    store.materialize(
        start_date=now - timedelta(days=10),
        end_date=now + timedelta(minutes=5),
    )

    print("\n--- Online Feature Retrieval for Real-time Inference ---")
    entity_rows: List[Dict[str, Any]] = [
        {
            "location_id": 1,
        },
    ]

    online_results = store.get_online_features(
        entity_rows=entity_rows,
        features=[
            "forecast_view:temp_lag_24",
        ],
    )

    print(f"Online features for location_id=1: {online_results}")

    print("\nForecasting workflow completed.")


if __name__ == "__main__":
    main()
