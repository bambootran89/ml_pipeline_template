"""
End-to-end demonstration of a recommendation system workflow using Feast
Feature Store (v0.58.0+).

This module simulates user–item interactions, initializes a Feast feature
repository, registers core entities and feature views, materializes features
from offline storage into an online store, and retrieves them for ranking or
serving RecSys models.

All timestamps are explicitly UTC-aware to guarantee correctness of
point-in-time joins and prevent schema or retrieval failures in Feast.
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
    """
    Generate a synthetic user–item interaction dataset stored as a Parquet file.

    The dataset includes interaction frequency features such as view counts and
    engagement ratios. It injects null values into selected rows to emulate
    real-world sparsity and validate the feature store behavior under missing
    data conditions.

    All generated timestamps are timezone-aware in UTC to ensure compatibility
    with Feast v0.58.0+ feature retrieval and materialization.

    Args:
        repo_path:
            Root path of the Feast feature repository where the interaction data
            will be written under the ``data`` subdirectory.
        hours:
            Number of hourly interaction records to simulate. Determines both the
            timestamp range and total row count.

    Returns:
        Absolute path to the generated ``interactions.parquet`` file.
    """
    file_path = repo_path / "data" / "interactions.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Use UTC timestamps to avoid join or retrieval failures in Feast
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
            "view_count": np.random.poisson(5, n_rows).astype(float),
            "like_ratio": np.random.uniform(0, 1, n_rows).astype(float),
        }
    )

    # Introduce missing values to simulate sparse engagement for testing
    df.loc[10:15, "like_ratio"] = np.nan
    df.to_parquet(file_path)
    return file_path.absolute()


def build_store(repo_name: str, data_file: Path) -> Any:
    """
    Initialize a Feast feature store and register RecSys entities and views.

    The function constructs a feature store using a factory abstraction,
    registers a user entity keyed by ``user_id``, and defines a feature view
    backed by the provided Parquet interaction dataset. It also configures
    a TTL to control staleness of online features.

    This function does not suppress exceptions so that schema or registration
    errors are surfaced during development and testing.

    Args:
        repo_name:
            Name of the Feast repository directory used to initialize the store.
        data_file:
            Absolute or relative path to the Parquet file serving as the source
            of the feature view.

    Returns:
        A configured Feast feature store instance with registered components.
    """
    store = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)

    print("Registering entity 'user' with join_key 'user_id'...")
    store.register_entity(
        name="user",
        join_key="user_id",
        description="Entity representing users for interaction-based joins",
        value_type="int",
    )

    print("Registering feature view 'recsys_view'...")
    store.register_feature_view(
        name="recsys_view",
        entities=["user"],
        schema={"view_count": "float", "like_ratio": "float", "item_id": "int"},
        source_path=str(data_file),
        ttl_days=2,
    )

    return store


def main() -> None:
    """
    Execute the complete RecSys feature store workflow.

    The workflow performs the following steps:
    1. Initialize the Feast repository structure on disk.
    2. Generate synthetic user–item interaction data.
    3. Build the feature store and register entities and feature views.
    4. Materialize offline features into the online store.
    5. Retrieve online features for ranking, personalization, or model serving.
    6. Print retrieved feature vectors for verification.

    No business logic or data transformation rules are modified here; the
    function only orchestrates feature store interactions for demo purposes.
    """
    repo_name = "recsys_repo"
    repo_path = Path(repo_name)

    print(f"--- Step 1: Initializing {repo_name} ---")
    FeastRepositoryManager.initialize_repo(repo_name)
    data_file = generate_interaction_data(repo_path)

    print("\n--- Step 2: Building Store ---")
    store = build_store(repo_name, data_file)

    print("\n--- Step 3: Materializing Features ---")
    now = datetime.now(timezone.utc)

    # Materialize a wide historical window and slightly into the future to ensure
    # latest interactions are included in the online store
    store.materialize(
        start_date=now - timedelta(days=7),
        end_date=now + timedelta(minutes=10),
    )

    print("\n--- Step 4: Online Retrieval ---")
    entity_rows: List[Dict[str, Any]] = [
        {"user_id": 1},
        {"user_id": 2},
        {"user_id": 3},
    ]

    online_results = store.get_online_features(
        entity_rows=entity_rows,
        features=["recsys_view:like_ratio", "recsys_view:view_count"],
    )

    print("Retrieved feature vectors for user ranking and personalization:")
    for result in online_results:
        print(f"  > {result}")

    print("\nRecSys feature store demo completed successfully.")


if __name__ == "__main__":
    main()
