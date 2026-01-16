"""Type definitions for API generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data type and features."""

    data_type: str
    features: List[str]
    input_chunk_length: int
    output_chunk_length: int
    path: str
    entity_key: str

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> DataConfig:
        """Create from dictionary with defaults."""
        if config is None:
            config = {}
        return cls(
            data_type=config.get("data_type", "timeseries"),
            features=config.get("features", []),
            input_chunk_length=config.get("input_chunk_length", 24),
            output_chunk_length=config.get("output_chunk_length", 6),
            path=config.get("path", ""),
            entity_key=config.get("entity_key", ""),
        )

    @property
    def is_feast(self) -> bool:
        """Check if data source is Feast."""
        return self.path.startswith("feast://")

    @property
    def feast_repo_path(self) -> str:
        """Extract Feast repository path from URI."""
        if not self.is_feast:
            return ""
        # Format: feast://path/to/repo?query
        return self.path.split("feast://")[1].split("?")[0]


@dataclass(frozen=True)
class GenerationContext:
    """Context for code generation."""

    pipeline_name: str
    load_map: Dict[str, str]
    preprocessor: Optional[Dict[str, Any]]
    inference_steps: List[Dict[str, Any]]
    experiment_config_path: str
    data_config: DataConfig
    model_keys: List[str]
    alias: str = "production"
