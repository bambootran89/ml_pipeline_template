"""
Engineer features from CSV and register them to Feast Feature Store.

This script loads a raw CSV dataset, derives lag/rolling/cyclical
features, writes them to a Parquet file, registers the entity and
feature view in Feast, and optionally materializes features to the
online store.

Usage:
    python mlproject.src.pipeline.populate_feast \\
        --csv <raw_csv_path> \\
        --repo <feast_repository_path> \\
        --entity <entity_join_key> \\
        --materialize
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

import pandas as pd

from mlproject.src.features.engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
)
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager

logger = logging.getLogger(__name__)


def _fail(msg: str) -> NoReturn:
    """Log error and exit."""
    logger.error(msg)
    raise SystemExit(1)


def main() -> None:
    """Main entry point for feature engineering and Feast registration."""
    parser = argparse.ArgumentParser(
        description="Engineer features from CSV and register to Feast"
    )
    parser.add_argument("--csv", required=True, help="Raw CSV file path")
    parser.add_argument("--repo", default="feature_repo", help="Feast repo path")
    parser.add_argument("--entity", default="location_id", help="Feast join key")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize features to online store",
    )
    args = parser.parse_args()

    # 1. Load raw data
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        _fail(f"Failed reading CSV: {e}")

    if "date" not in df.columns:
        _fail("CSV must contain a 'date' column")

    df["event_timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop(columns=["date"])

    # Add entity column
    df[args.entity] = 1

    # 2. Feature engineering
    print("Engineering features...")
    try:
        df = add_lag_features(df, ["HUFL", "MUFL"], lags=[24])
        df = add_rolling_features(df, ["HUFL"], windows=[12], agg="mean")
        df = add_time_features(df, "event_timestamp")
    except Exception as e:
        _fail(f"Feature engineering failed: {e}")

    df = df.dropna()

    # 3. Save features to Parquet
    try:
        FeastRepositoryManager.initialize_repo(args.repo)
    except OSError as e:
        _fail(f"Feast repo init failed: {e}")

    data_dir = Path(args.repo) / "data"
    data_dir.mkdir(exist_ok=True)

    parquet_path = data_dir / "features.parquet"
    df.to_parquet(parquet_path)

    print(f"Saved {len(df)} rows to {parquet_path}")

    # 4. Register entity & feature view in Feast
    try:
        store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=args.repo,
        )
    except Exception as e:
        _fail(f"Failed initializing Feast store: {e}")

    store.register_entity(
        name="location",
        join_key=args.entity,
        value_type="int",
    )

    schema = {
        "HUFL": "float",
        "MUFL": "float",
        "HUFL_lag24": "float",
        "MUFL_lag24": "float",
        "HUFL_roll12_mean": "float",
        "hour_sin": "float",
        "hour_cos": "float",
        "dow_sin": "float",
        "dow_cos": "float",
    }

    store.register_feature_view(
        name="etth1_features",
        entities=["location"],
        schema=schema,
        source_path=str(parquet_path.absolute()),
        ttl_days=365,
    )

    print("Registered feature view 'etth1_features'")

    # 5. Optional: materialize to online store
    if args.materialize:
        datetime.now(timezone.utc)
        start = df["event_timestamp"].min()
        end = df["event_timestamp"].max()

        # Safety check for timezone awareness
        if start.tzinfo is None or end.tzinfo is None:
            _fail("Timestamps must be UTC-aware before materialization")

        store.materialize(start, end)
        print(f"Materialized {start.date()} → {end.date()}")

    print("Feature population workflow completed.")


if __name__ == "__main__":
    main()
