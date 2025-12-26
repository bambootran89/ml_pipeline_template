"""
Demonstration of a fraud detection feature store workflow.

This module shows how to generate synthetic fraud transaction data,
initialize a Feast-based feature repository, register entities and
feature views, and retrieve features in both offline and online modes.

The implementation is compatible with Feast v0.58.0+ and enforces
UTC-aware timestamps to ensure correctness of point-in-time joins.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager


def generate_fraud_data(repo: Path, n: int = 5000) -> Path:
    """
    Generate a synthetic fraud transaction dataset with UTC timestamps.

    The generated dataset simulates user transactions with multiple
    risk-related features, including transaction amount, recent
    transaction count, device risk, and a derived risk score.

    All timestamps are explicitly timezone-aware (UTC) to satisfy
    Feast v0.58.0+ requirements for reliable point-in-time joins.

    Args:
        repo:
            Root path of the feature repository where the data file
            will be stored under the ``data`` directory.
        n:
            Number of transaction records to generate.

    Returns:
        Absolute path to the generated Parquet file containing the
        synthetic fraud transaction data.
    """
    file_path = repo / "data" / "fraud.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Feast v0.58.0+ requires UTC-aware timestamps for reliable joins
    now = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=n // 60 + 1),
        periods=n,
        freq="min",
        tz=timezone.utc,
    )

    df = pd.DataFrame(
        {
            "user_id": np.random.randint(1, 50, size=n).astype(int),
            "event_timestamp": ts,
            "amount": np.random.exponential(scale=200, size=n).astype(float),
            "tx_count_30m": np.random.poisson(3, size=n).astype(float),
            "device_risk": np.random.rand(n).astype(float),
        }
    )

    df["high_amount_flag"] = (df["amount"] > 1000).astype(float)
    df["risk_score"] = (
        0.6 * df["device_risk"]
        + 0.4 * (df["tx_count_30m"] / 10)
        + 0.2 * df["high_amount_flag"]
    ).clip(0, 1)

    # Ensure no naive timestamps are written to storage
    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """
    Create and configure a Feast feature store for fraud detection.

    This function initializes a Feast-backed feature store, registers
    the required entity, and defines a feature view backed by the
    provided Parquet data source.

    No exceptions are silently suppressed in order to surface schema,
    configuration, or registration errors during development.

    Args:
        repo_name:
            Name of the Feast repository directory.
        data_file:
            Path to the Parquet file used as the feature view source.

    Returns:
        An initialized feature store instance with registered entities
        and feature views.
    """
    store = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)

    print("--- Registering Entity ---")
    store.register_entity(
        name="user",
        join_key="user_id",
        description="User identifier used for fraud feature joins",
        value_type="int",
    )

    print("--- Registering Feature View ---")
    store.register_feature_view(
        name="fraud_view",
        entities=["user"],
        schema={
            "amount": "float",
            "device_risk": "float",
            "risk_score": "float",
            "high_amount_flag": "float",
        },
        source_path=str(data_file),
        ttl_days=1,
    )

    return store


def main() -> None:
    """
    Run the end-to-end fraud detection feature store demo.

    The workflow includes:
    1. Initializing a Feast repository.
    2. Generating synthetic fraud transaction data.
    3. Registering entities and feature views.
    4. Retrieving historical (offline) features.
    5. Materializing features to the online store.
    6. Retrieving features from the online store.
    """
    repo_name = "fraud_repo"
    repo = Path(repo_name)

    # Step 1: Initialize Repo
    print(f"Step 1: Initializing {repo_name}...")
    FeastRepositoryManager.initialize_repo(repo_name)
    data_file = generate_fraud_data(repo)

    # Step 2: Build Store
    print("Step 2: Building Feature Store...")
    store = build_store(repo_name, data_file)

    # Step 3: Historical Retrieval (Offline)
    print("Step 3: Fetching Historical Features...")
    # Use UTC timestamps to match the source data timezone
    now = datetime.now(timezone.utc)
    entity_df = pd.DataFrame(
        {
            "user_id": [5, 10, 20],
            "event_timestamp": [
                now - timedelta(minutes=1),
                now - timedelta(minutes=5),
                now - timedelta(minutes=10),
            ],
        }
    )

    hist = store.get_historical_features(
        entity_df, ["fraud_view:risk_score", "fraud_view:amount"]
    )

    print("\n--- Historical Features Result ---")
    print(hist.head())

    # Step 4: Materialization (Offline -> Online)
    print("\nStep 4: Materializing to Online Store...")
    store.materialize(now - timedelta(days=1), now + timedelta(minutes=5))

    # Step 5: Online Retrieval
    print("Step 5: Fetching Online Features...")
    online = store.get_online_features(
        [{"user_id": 5}, {"user_id": 10}],
        ["fraud_view:risk_score", "fraud_view:amount"],
    )

    print("\n--- Online Features Result ---")
    for row in online:
        print(row)


if __name__ == "__main__":
    main()
