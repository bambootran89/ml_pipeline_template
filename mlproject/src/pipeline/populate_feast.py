"""Feature engineering pipeline that generates time-series features and registers them
to a Feast Feature Store.

The pipeline performs:
1) Load raw data from CSV.
2) Derive lag, rolling, and cyclical time features.
3) Drop invalid rows and persist the result to Parquet.
4) Initialize/ensure a Feast repository exists.
5) Register the entity and feature view in Feast.
6) Optionally materialize the feature data to the online store.

CLI usage:
    python -m mlproject.src.pipeline.populate_feast \\
        --csv <path_to_raw_csv> \\
        --repo <feast_repo_path> \\
        --entity <join_key_column> \\
        --materialize
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

from mlproject.src.features.engineering import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
)
from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.repository import FeastRepositoryManager

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _fail(msg: str) -> NoReturn:
    """Log an error message and terminate the program with exit code 1."""
    logger.error(msg)
    raise SystemExit(1)


def _load_raw(csv_path: str, entity_col: str) -> pd.DataFrame:
    """Load the raw CSV and prepare a UTC-aware event_timestamp and entity column."""
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        _fail("Input CSV must contain a 'date' column")

    df["event_timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop(columns=["date"])
    df[entity_col] = 1

    logger.info(
        "Raw data loaded from CSV with shape: (%d, %d)",
        df.shape[0],
        df.shape[1],
    )
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lag, rolling, and time feature transformations and drop invalid rows."""
    df = add_lag_features(df, ["HUFL", "MUFL"], lags=[24])
    df = add_rolling_features(df, ["HUFL", "MUFL"], windows=[12], agg="mean")
    df = add_time_features(df, "event_timestamp")
    df = df.dropna()

    logger.info(
        "Engineered feature data generated with shape: (%d, %d)",
        df.shape[0],
        df.shape[1],
    )
    return df


def _save(df: pd.DataFrame, repo_path: str) -> Path:
    """Ensure Feast repo exists, create data dir,
    and save the feature data to Parquet."""
    FeastRepositoryManager.initialize_repo(repo_path)

    data_dir = Path(repo_path) / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "features.parquet"

    df.to_parquet(out_path)

    logger.info("Feature data persisted to Parquet at: %s", str(out_path))
    return out_path


def _register(store: Any, entity_col: str, source: Path) -> None:
    """Register the entity and feature view in Feast using the Parquet schema."""
    store.register_entity(
        name="location",
        join_key=entity_col,
        value_type="int",
    )

    schema = {
        "HUFL": "float",
        "MUFL": "float",
        "HUFL_lag24": "float",
        "MUFL_lag24": "float",
        "HUFL_roll12_mean": "float",
        "MUFL_roll12_mean": "float",
        "hour_sin": "float",
        "hour_cos": "float",
        "dow_sin": "float",
        "dow_cos": "float",
    }

    store.register_feature_view(
        name="etth1_features",
        entities=["location"],
        schema=schema,
        source_path=str(source.absolute()),
        ttl_days=365,
    )

    logger.info(
        "Feast entity and feature view 'etth1_features' registered successfully"
    )


def _materialize_full(store: Any, df: pd.DataFrame) -> None:
    """Materialize feature data to the Feast online store with
    UTC-aware safety checks."""
    start = df["event_timestamp"].min()
    end = df["event_timestamp"].max()

    if start.tzinfo is None or end.tzinfo is None:
        _fail("Timestamps must be UTC-aware before materialization")

    store.materialize(start, end)

    logger.info(
        "Feature data materialized to online store for range: %s → %s",
        str(start.date()),
        str(end.date()),
    )


def main() -> None:
    """Orchestrate the full feature engineering and Feast registration pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate and register time-series features to Feast"
    )
    parser.add_argument("--csv", required=True, help="Path to the raw input CSV file")
    parser.add_argument("--repo", default="feature_repo", help="Path to Feast repo")
    parser.add_argument("--entity", default="location_id", help="Feast join key column")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize features to the online store",
    )
    args = parser.parse_args()

    try:
        df = _load_raw(args.csv, args.entity)
        df = _engineer(df)

        feat_source = _save(df, args.repo)

        store = FeatureStoreFactory.create(
            store_type="feast",
            repo_path=args.repo,
        )
        _register(store, args.entity, feat_source)

        if args.materialize:
            _materialize_full(store, df)

        logger.info(
            "Pipeline completed without errors at %s", str(datetime.now(timezone.utc))
        )
        print("Feature population pipeline completed successfully")

    except Exception as err:
        _fail(f"Pipeline failed: {err}")


if __name__ == "__main__":
    main()
