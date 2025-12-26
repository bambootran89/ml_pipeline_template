"""
Feast Feature Store repository bootstrap and configuration management for
Feast v0.58.0+.

This module provides ``FeastRepositoryManager`` for initializing a minimal
Feast repository structure on disk and generating a deterministic
``feature_store.yaml`` configuration file using:
- Relative paths for registry and online store persistence.
- Serialization version 3 for entity key encoding.
- Explicit offline/online store type declarations.

The implementation avoids dynamic path guessing and ensures that the
repository layout is ready for PIT (point-in-time) compatible feature views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class FeastRepositoryManager:
    """
    Utility class for creating and managing Feast repository configuration files.

    This class exposes static methods to bootstrap a Feast feature repository
    directory, prepare the required ``data`` subdirectory, and generate a
    ``feature_store.yaml`` configuration file with deterministic, relative paths.

    The class must not be instantiated and only exposes static methods.
    """

    @staticmethod
    def initialize_repo(repo_path: str = "feature_repo") -> None:
        """
        Create a Feast repository directory and write a ``feature_store.yaml`` file.

        The generated configuration ensures:
        - Relative path persistence for ``registry`` and ``online_store.path``.
        - ``entity_key_serialization_version`` set to 3 for consistent entity encoding.
        - Local provider mode for development and testing.
        - File-based offline store declaration for batch PIT joins.
        - SQLite-based online store for inference-time lookups.

        Args:
            repo_path:
                Filesystem path to the Feast repository root directory.

        Raises:
            OSError:
                If the directory or configuration file cannot be created.
        """
        base_path = Path(repo_path)
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / "data").mkdir(exist_ok=True)

        # Ensure the offline source directory exists for feature view ingestion.
        (base_path / "data").mkdir(exist_ok=True)

        config: Dict[str, Any] = {
            "project": "mlproject",
            "registry": "data/registry.db",
            "provider": "local",
            "entity_key_serialization_version": 3,
            "offline_store": {"type": "file"},
            "online_store": {"type": "sqlite", "path": "data/online.db"},
        }

        with open(base_path / "feature_store.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f)
