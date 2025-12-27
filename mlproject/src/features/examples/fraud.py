"""
Fraud detection feature store workflow demo.

This module generates synthetic fraud transaction data, initializes
a Feast-compatible repository, registers entities and feature views,
materializes offline features into the online store, and retrieves
fraud features for model training and serving using UTC timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager


def generate_fraud_data(repo: Path, n: int = 5000) -> Path:
    """Generate synthetic fraud transaction data stored as a UTC-aware Parquet file."""
    file_path = repo / "data" / "fraud.parquet"
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=n // 60 + 1),
        periods=n,
        freq="min",
        tz=timezone.utc,
    )

    df = pd.DataFrame(
        {
            "user_id": np.random.randint(
                1,
                50,
                size=n,
            ).astype(int),
            "event_timestamp": ts,
            "amount": np.random.exponential(
                scale=200,
                size=n,
            ).astype(float),
            "tx_count_30m": np.random.poisson(
                3,
                size=n,
            ).astype(float),
            "device_risk": np.random.rand(n).astype(float),
        }
    )

    df["high_amount_flag"] = (df["amount"] > 1000).astype(float)
    df["risk_score"] = (
        0.6 * df["device_risk"]
        + 0.4 * (df["tx_count_30m"] / 10)
        + 0.2 * df["high_amount_flag"]
    ).clip(
        0,
        1,
    )

    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """Initialize a Feast feature store and register fraud detection components."""
    store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    print("--- Registering entity for fraud joins ---")
    store.register_entity(
        name="user",
        join_key="user_id",
        description="User identifier for fraud feature joins",
        value_type="int",
    )

    print("--- Registering fraud feature view ---")
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
    """Execute the end-to-end fraud feature store workflow demo."""
    repo_name = "fraud_repo"
    repo = Path(repo_name)

    print(f"Step 1: Initializing repository '{repo_name}'...")
    FeastRepositoryManager.initialize_repo(repo_name)
    data_file = generate_fraud_data(repo)

    print("Step 2: Building feature store configuration...")
    store = build_store(repo_name, data_file)

    print("Step 3: Retrieving offline historical fraud features...")
    now = datetime.now(timezone.utc)

    entity_df: pd.DataFrame = pd.DataFrame(
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
        entity_df,
        features=[
            "fraud_view:risk_score",
            "fraud_view:amount",
        ],
    )

    print("\n--- Historical Fraud Features (Offline) ---")
    print(hist.head())

    print("\nStep 4: Materializing features to the online store...")
    store.materialize(
        start_date=now - timedelta(days=1),
        end_date=now + timedelta(minutes=5),
    )

    print("Step 5: Retrieving fraud features from the online store...")
    entity_rows: List[Dict[str, Any]] = [
        {"user_id": 5},
        {"user_id": 10},
    ]

    online = store.get_online_features(
        entity_rows=entity_rows,
        features=[
            "fraud_view:risk_score",
            "fraud_view:amount",
        ],
    )

    print("\n--- Fraud Features (Online Store) ---")
    for row in online:
        print(f"  > {row}")

    print("\nFraud feature store workflow demo completed.")


if __name__ == "__main__":
    main()
