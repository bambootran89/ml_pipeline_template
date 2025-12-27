"""
Feature engineering and population pipeline for Feast Feature Store.

This module loads a raw CSV dataset, derives time-series lag, rolling, and
cyclical calendar features, writes them to Parquet, registers the entity and
FeatureView to a Feast repository, and optionally materializes features to the
online store for real-time inference.

All timestamps must be timezone-aware (UTC) to guarantee deterministic
point-in-time (PIT) joins and consistent online materialization behavior.

CLI Usage:
    python -m mlproject.src.pipeline.populate_feast \
        --csv <raw_csv_path> \
        --repo <feast_repository_path> \
        --entity <entity_join_key> \
        --materialize
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, NoReturn, Tuple, Union

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


def _fail(message: str) -> NoReturn:
    """
    Log an error message and terminate execution immediately.

    Args:
        message: Human-readable error explanation.

    Raises:
        SystemExit: Always exits the program with status code 1.
    """
    logger.error(message)
    raise SystemExit(1)


def _parse_uri(uri: str) -> Tuple[str, str, Union[int, str], str]:
    """
    Parse a Feast historical retrieval URI.

    Expected format:
        feast://<repo_path>?entity=<key>&id=<val>&features=<list>

    Args:
        uri: A Feast URI string.

    Returns:
        A tuple containing:
            repo_path: Feast repository root path.
            entity_key: Entity join key used for PIT lookups.
            entity_id: Default entity identifier for single-series retrieval.
            features: Comma-separated list of features.

    Raises:
        ValueError: If the URI does not follow the required prefix or is malformed.
    """
    if not uri.startswith("feast://"):
        raise ValueError("Feast URI must start with 'feast://'")

    body = uri[8:]
    repo_path, _, query = body.partition("?")

    params: dict[str, str] = {}
    for pair in query.split("&"):
        if not pair or "=" not in pair:
            continue
        key, _, val = pair.partition("=")
        params[key] = val

    entity_key = params.get("entity", "")
    entity_id: Union[int, str] = params.get("id", "1")
    features = params.get("features", "")

    if not entity_key or not features:
        raise ValueError("Feast URI missing required 'entity' or 'features' params")

    try:
        entity_id = int(entity_id)
    except ValueError:
        pass

    return repo_path, entity_key, entity_id, features


def _load_raw(csv_path: str, entity_key: str) -> pd.DataFrame:
    """
    Load a raw CSV file and normalize its timestamp and entity column.

    Args:
        csv_path: Filesystem path to the raw CSV file.
        entity_key: Name of the entity join key to insert if not present.

    Returns:
        DataFrame with an 'event_timestamp' column (UTC-aware) and entity column.

    Raises:
        SystemExit: If the CSV cannot be read or lacks a 'date' column.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        _fail(f"Failed reading CSV: {exc}")

    if "date" not in df.columns:
        _fail("Input CSV must contain a 'date' column for timestamp parsing")

    df["event_timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop(columns=["date"])

    df[entity_key] = 1
    logger.info("Raw data loaded with shape: %s", df.shape)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-series lag, rolling, and cyclical calendar features.

    The function preserves all input columns while generating new
    engineered features required for model training and inference.

    Args:
        df: Input DataFrame containing UTC timestamps.

    Returns:
        DataFrame with added engineered feature columns, NaNs removed.
    """
    print("Engineering features...")
    try:
        df = add_lag_features(df, ["HUFL", "MUFL"], lags=[24])
        df = add_rolling_features(df, ["HUFL", "MUFL"], windows=[12], agg="mean")
        df = add_time_features(df, "event_timestamp")
    except Exception as exc:
        _fail(f"Feature engineering stage failed: {exc}")

    df = df.dropna()
    logger.info("Engineered features generated with shape: %s", df.shape)
    return df


def _store_offline_features(df: pd.DataFrame, repo: str) -> Path:
    """
    Initialize a Feast repository on disk and write features to Parquet.

    The Parquet file becomes the offline source for PIT historical retrieval
    and batch materialization into the online store.

    Args:
        df: DataFrame containing engineered features and UTC timestamps.
        repo: Feast repository root path.

    Returns:
        Absolute path to the stored 'features.parquet' file.

    Raises:
        SystemExit: If the repository cannot be initialized or file cannot be written.
    """
    try:
        FeastRepositoryManager.initialize_repo(repo)
    except OSError as exc:
        _fail(f"Feast repository initialization failed: {exc}")

    data_dir = Path(repo) / "data"
    data_dir.mkdir(exist_ok=True)

    parquet_path = data_dir / "features.parquet"
    df.to_parquet(parquet_path)

    logger.info("Offline features stored with shape: %s", df.shape)
    print(f"Saved {len(df)} rows to {parquet_path}")
    return parquet_path.absolute()


def _register_to_feast(
    parquet_path: Path,
    repo: str,
    entity_key: str,
) -> None:
    """
    Register a PIT-compatible entity and FeatureView to Feast.

    The FeatureView schema must match the columns stored in the Parquet
    offline source to ensure successful registration and materialization.

    Args:
        parquet_path: Absolute path to the offline features Parquet file.
        repo: Feast repository root path.
        entity_key: Entity join key for PIT joins.

    Raises:
        SystemExit: If the Feast store cannot be initialized.
    """
    try:
        store = FeatureStoreFactory.create(store_type="feast", repo_path=repo)
    except Exception as exc:
        _fail(f"Failed initializing Feast store: {exc}")

    store.register_entity(
        name="location",
        join_key=entity_key,
        value_type="int",
        description="Auto-registered time-series entity for PIT lookups",
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
        source_path=str(parquet_path),
        ttl_days=365,
    )

    print("Registered FeatureView 'etth1_features'")
    logger.info("Entity and FeatureView successfully registered to Feast")


def _materialize_full(store: Any, df: pd.DataFrame) -> None:
    """
    Materialize the full offline timestamp range into the online store.

    This step allows real-time feature vector retrieval without requiring
    a local input file at inference time.

    Args:
        store: Feast store instance or wrapper.
        df: DataFrame containing timezone-aware UTC timestamps.

    Raises:
        SystemExit: If timestamps are not UTC-aware.
    """
    start = df["event_timestamp"].min()
    end = df["event_timestamp"].max()

    if start.tzinfo is None or end.tzinfo is None:
        _fail("Timestamps must be timezone-aware (UTC) for materialization")

    store.materialize(start, end)
    print(f"Materialized {start.date()} -> {end.date()}")
    logger.info("Materialization completed for range: %s -> %s", start, end)


def main() -> None:
    """
    Bootstrap a Feast feature population workflow from a raw CSV dataset.

    The function performs:
    1. Raw data loading and timestamp normalization.
    2. Time-series feature engineering (lag, rolling, cyclical time).
    3. Offline feature persistence to Parquet for PIT joins.
    4. Entity and FeatureView registration in Feast.
    5. Optional online feature materialization for inference.

    The function maintains a high-level orchestration pattern without embedding
    transformation logic directly, ensuring lower cyclomatic complexity and
    improved maintainability.
    """
    parser = argparse.ArgumentParser(
        description="Populate a Feast repository with engineered features"
    )
    parser.add_argument("--csv", required=True, help="Raw CSV path")
    parser.add_argument("--repo", default="feature_repo", help="Feast repo path")
    parser.add_argument("--entity", default="location_id", help="Entity join key")
    parser.add_argument("--materialize", action="store_true")

    args = parser.parse_args()

    df = _load_raw(args.csv, args.entity)
    df = _engineer_features(df)
    parquet_path = _store_offline_features(df, args.repo)
    _register_to_feast(parquet_path, args.repo, args.entity)

    if args.materialize:
        _materialize_full(df, args.repo)

    print("Feature population pipeline completed successfully")
    logger.info("Pipeline execution completed")
