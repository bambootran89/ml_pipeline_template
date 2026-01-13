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
        )


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
