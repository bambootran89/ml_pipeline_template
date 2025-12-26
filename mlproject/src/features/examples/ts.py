"""
End-to-end demonstration of a time-series forecasting workflow using Feast
Feature Store (v0.58.0+).

This script simulates sensor telemetry for forecasting, registers entities and
feature views, performs offline point-in-time historical feature retrieval,
materializes features to an online store, and reads back the latest online
feature values for real-time inference.

All timestamps are explicitly timezone-aware in UTC to ensure deterministic PIT
joins and reliable materialization behavior in Feast v0.58.0+.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager


def generate_ts_data(repo: Path, periods: int = 200) -> Path:
    """
    Generate synthetic time-series sensor data and precompute lag and rolling features.

    The generated dataset simulates a single sensor location (``location_id``),
    emits continuous hourly telemetry, derives lag and rolling features in UTC,
    and writes the result to disk as a Parquet file for Feast ingestion.

    Derived features:
    - ``temp_lag_24``: 24-hour lag of temperature.
    - ``demand_roll_12``: 12-hour rolling mean of demand.

    Args:
        repo:
            Root directory of the Feast repository where data will be written
            under the ``data`` subdirectory.
        periods:
            Number of hourly time steps to generate.

    Returns:
        Absolute path to the generated ``ts_features.parquet`` file.
    """
    file_path = repo / "data" / "ts_features.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Use UTC to ensure compatibility with Feast point-in-time joins.
    now = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=periods),
        periods=periods,
        freq="H",
        tz=timezone.utc,
    )

    # ``location_id`` is introduced as a proper entity for realistic TS joins.
    df = pd.DataFrame(
        {
            "location_id": [1] * periods,
            "event_timestamp": ts,
            "temperature": np.random.randn(periods).astype(float) * 10 + 25,
            "demand": np.random.randn(periods).astype(float) * 100 + 500,
        }
    )

    # Precompute lag and rolling window features before storing for Feast.
    df["temp_lag_24"] = df["temperature"].shift(24)
    df["demand_roll_12"] = df["demand"].rolling(12).mean()

    # Write to Parquet for offline feature view source.
    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """
    Initialize a Feast feature store and register time-series entities and views.

    The store is constructed via a factory abstraction and configured without
    silently suppressing errors to ensure schema or registration failures are
    surfaced during development.

    Registered components:
    - Entity: ``location`` joined on ``location_id``.
    - Feature view: ``forecast_view`` backed by the generated Parquet source.

    Args:
        repo_name:
            Name of the Feast repository directory to initialize the store.
        data_file:
            Path to the Parquet file used as the source
            for the time-series feature view.

    Returns:
        Configured feature store instance with registered entities and feature views.
    """
    store = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)

    print("Registering entity 'location'...")
    store.register_entity(
        name="location",
        join_key="location_id",
        description="Entity representing sensor locations for time-series joins",
        value_type="int",
    )

    print("Registering feature view 'forecast_view'...")
    store.register_feature_view(
        name="forecast_view",
        entities=["location"],
        schema={"temp_lag_24": "float", "demand_roll_12": "float"},
        source_path=str(data_file),
        ttl_days=7,
    )

    return store


def main() -> None:
    """
    Orchestrate the complete time-series forecasting feature store workflow.

    Steps executed:
    1. Initialize Feast repository structure.
    2. Generate synthetic time-series data.
    3. Build store and register entity + feature views.
    4. Retrieve offline historical features using PIT semantics.
    5. Materialize features to the online store.
    6. Retrieve latest online features for real-time inference.
    7. Print feature vectors for verification.

    This function modifies no forecasting or transformation logic — it only
    coordinates interactions with the feature store for demonstration purposes.
    """
    repo_name = "ts_repo"
    repo = Path(repo_name)

    # Step 1: Initialize repository structure for Feast.
    FeastRepositoryManager.initialize_repo(repo_name)
    data_file = generate_ts_data(repo)

    # Step 2: Build the store and register forecasting components.
    store = build_store(repo_name, data_file)

    # Step 3: Perform offline point-in-time historical feature retrieval.
    print("\n--- Step 3: Performing Point-in-Time Historical Feature Retrieval ---")
    now = datetime.now(timezone.utc)
    entity_df = pd.DataFrame(
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
        features=["forecast_view:temp_lag_24", "forecast_view:demand_roll_12"],
    )
    print("Historical feature sample (first 5 rows):")
    print(hist_df.head())

    # Step 4: Materialize a broad historical window to the online store.
    print("\n--- Step 4: Materializing Features to Online Store ---")
    store.materialize(
        start_date=now - timedelta(days=10),
        end_date=now + timedelta(minutes=5),
    )

    # Step 5: Retrieve the latest online feature values for real-time inference.
    print("\n--- Step 5: Online Feature Retrieval ---")
    online_results = store.get_online_features(
        entity_rows=[{"location_id": 1}],
        features=["forecast_view:temp_lag_24"],
    )

    print(f"Online features for location_id=1: {online_results}")

    print("\nTime-series forecasting workflow completed.")


if __name__ == "__main__":
    main()
