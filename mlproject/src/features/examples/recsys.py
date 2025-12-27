"""
Recommendation system workflow demo using Feast Feature Store.

This module simulates user–item interactions, initializes a Feast
repository, registers entities and feature views, materializes offline
features into an online store, and retrieves them for ranking or serving
RecSys models with UTC-aware timestamps.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager


def generate_interaction_data(repo_path: Path, hours: int = 72) -> Path:
    """Generate synthetic user–item interactions stored as a Parquet file."""
    file_path = repo_path / "data" / "interactions.parquet"
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    now = datetime.now(timezone.utc)
    ts = pd.date_range(
        start=now - timedelta(hours=hours),
        periods=hours,
        freq="H",
        tz=timezone.utc,
    )

    n_rows = len(ts)
    df = pd.DataFrame(
        {
            "user_id": (np.arange(n_rows) % 4 + 1).astype(int),
            "item_id": (np.arange(n_rows) % 10 + 101).astype(int),
            "event_timestamp": ts,
            "view_count": np.random.poisson(
                5,
                n_rows,
            ).astype(float),
            "like_ratio": np.random.uniform(
                0,
                1,
                n_rows,
            ).astype(float),
        }
    )

    df.loc[10:15, "like_ratio"] = np.nan
    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """Initialize Feast store and register RecSys entities and feature views."""
    store = FeatureStoreFactory.create(
        store_type="feast",
        repo_path=repo_name,
    )

    print("Registering entity 'user' with join key 'user_id'...")
    store.register_entity(
        name="user",
        join_key="user_id",
        description="Users for interaction-based joins",
        value_type="int",
    )

    print("Registering feature view 'recsys_view'...")
    store.register_feature_view(
        name="recsys_view",
        entities=["user"],
        schema={
            "view_count": "float",
            "like_ratio": "float",
            "item_id": "int",
        },
        source_path=str(data_file),
        ttl_days=2,
    )

    return store


def main() -> None:
    """Orchestrate the end-to-end RecSys feature store workflow."""
    repo_name = "recsys_repo"
    repo_path = Path(repo_name)

    print("--- Initializing repository on disk ---")
    FeastRepositoryManager.initialize_repo(repo_name)

    data_file = generate_interaction_data(repo_path)

    print("\n--- Building feature store ---")
    store = build_store(repo_name, data_file)

    print("\n--- Materializing features into the online store ---")
    now = datetime.now(timezone.utc)

    store.materialize(
        start_date=now - timedelta(days=7),
        end_date=now + timedelta(minutes=10),
    )

    print("\n--- Retrieving online features ---")
    entity_rows: List[Dict[str, Any]] = [
        {"user_id": 1},
        {"user_id": 2},
        {"user_id": 3},
    ]

    online_results = store.get_online_features(
        entity_rows=entity_rows,
        features=[
            "recsys_view:like_ratio",
            "recsys_view:view_count",
        ],
    )

    print("Retrieved feature vectors for ranking and personalization:")
    for result in online_results:
        print(f"  > {result}")

    print("\nRecSys workflow demo completed.")


if __name__ == "__main__":
    main()
