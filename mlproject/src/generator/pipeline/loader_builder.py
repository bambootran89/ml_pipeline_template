"""MLflow loader builder utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


class LoaderBuilder:
    """Builds MLflow loader configuration."""

    def add_mlflow_loader(
        self,
        steps: List[Any],
        init_id: str,
        alias: str,
        preprocessors: List[Any],
        producers: List[Any],
        legacy_preprocessor: Optional[Any],
    ) -> None:
        """Add MLflow loader step to pipeline.

        Args:
            steps: List to append loader step.
            init_id: Initialization step ID.
            alias: Model alias for loading.
            preprocessors: List of preprocessor steps.
            producers: List of model producer steps.
            legacy_preprocessor: Legacy preprocessor if exists.
        """
        load_map = self._build_load_map(producers, preprocessors, legacy_preprocessor)

        steps.append(
            OmegaConf.create(
                {
                    "id": init_id,
                    "type": "mlflow_loader",
                    "enabled": True,
                    "alias": alias,
                    "load_map": load_map,
                }
            )
        )

    def _build_load_map(
        self,
        producers: List[Any],
        preprocessors: List[Any],
        legacy_preprocessor: Optional[Any],
    ) -> List[Dict[str, str]]:
        """Build load_map for MLflow loader.

        Args:
            producers: List of model producer steps.
            preprocessors: List of preprocessor steps.
            legacy_preprocessor: Legacy preprocessor if exists.

        Returns:
            List of load map entries.
        """
        load_map = [
            {"step_id": p.id, "context_key": f"fitted_{p.id}"} for p in producers
        ]

        for prep in preprocessors:
            load_map.append({"step_id": prep.id, "context_key": f"fitted_{prep.id}"})

        if not legacy_preprocessor:
            return load_map

        if any(p.id == legacy_preprocessor.id for p in preprocessors):
            return load_map

        load_map.append(
            {
                "step_id": legacy_preprocessor.id,
                "context_key": "transform_manager",
            }
        )

        return load_map
