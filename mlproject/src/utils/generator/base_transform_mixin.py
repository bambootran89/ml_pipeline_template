from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


class BaseTransformMixin:
    """
    Mixin providing core utility methods for pipeline configuration transformation.

    This class contains shared helper functions used during both Evaluation and
    Serving configuration generation. Its main responsibilities include:
    - Resetting step state for non-training execution (is_train=False).
    - Removing training-only parameters and artifacts (e.g. hyperparams).
    - Preparing wiring configuration for clustering models.
    - Managing load_map construction for loading trained models and
    preprocessors from MLflow.

    """

    MODEL_PRODUCER_TYPES = ["trainer", "clustering", "framework_model"]

    def _is_model_producer(self, step: Any) -> bool:
        """Check if step is a model producer."""
        if hasattr(step, "type"):
            if step.type in self.MODEL_PRODUCER_TYPES:
                return True
            # Support dynamic_adapter with artifact_type == "model"
            if step.type == "dynamic_adapter":
                return hasattr(step, "artifact_type") and step.artifact_type == "model"
        return False

    def _extract_from_sub_pipeline(self, step: Any) -> List[Any]:
        """Extract model producers from sub-pipeline."""
        if hasattr(step, "pipeline") and hasattr(step.pipeline, "steps"):
            return self._extract_model_producers_recursive(step.pipeline.steps)
        return []

    def _extract_from_branch(self, step: Any) -> List[Any]:
        """Extract model producers from branch if_true/if_false."""
        producers = []
        for branch_name in ["if_true", "if_false"]:
            if hasattr(step, branch_name):
                branch_step = getattr(step, branch_name)
                if self._is_model_producer(branch_step):
                    producers.append(branch_step)
        return producers

    def _extract_from_parallel(self, step: Any) -> List[Any]:
        """Extract model producers from parallel branches."""
        producers = []
        if hasattr(step, "branches"):
            for branch in step.branches:
                if self._is_model_producer(branch):
                    producers.append(branch)
        return producers

    def _extract_model_producers_recursive(self, steps: List[Any]) -> List[Any]:
        """Recursively extract model producers from steps."""
        producers: List[Any] = []
        for step in steps:
            if self._is_model_producer(step):
                producers.append(step)
            elif step.type == "sub_pipeline":
                producers.extend(self._extract_from_sub_pipeline(step))
            elif step.type == "branch":
                producers.extend(self._extract_from_branch(step))
            elif step.type == "parallel":
                producers.extend(self._extract_from_parallel(step))
        return producers

    def _extract_preprocessors_recursive(self, steps: List[Any]) -> List[Any]:
        """Recursively extract preprocessor steps."""
        preprocessors: List[Any] = []
        for step in steps:
            if step.type == "preprocessor":
                preprocessors.append(step)
            elif step.type == "dynamic_adapter":
                if (
                    hasattr(step, "artifact_type")
                    and step.artifact_type == "preprocess"
                ):
                    preprocessors.append(step)
            elif step.type == "sub_pipeline":
                if hasattr(step, "pipeline") and hasattr(step.pipeline, "steps"):
                    nested = self._extract_preprocessors_recursive(step.pipeline.steps)
                    preprocessors.extend(nested)
        return preprocessors

    def _setup_step_for_load_mode(self, step: Any, alias: str) -> None:
        """Setup step for eval/serve load mode."""
        step.is_train = False
        step.alias = alias
        step.instance_key = f"fitted_{step.id}"

    def _remove_training_configs(self, step: Any) -> None:
        """Remove training-only configuration attributes."""
        for attr in ["log_artifact", "artifact_type", "hyperparams"]:
            if hasattr(step, attr):
                delattr(step, attr)

    def _setup_clustering_wiring(self, step: Any) -> None:
        """Setup wiring for clustering step in eval/serve mode."""
        if not hasattr(step, "wiring"):
            step.wiring = OmegaConf.create({})
        if not hasattr(step.wiring, "inputs"):
            step.wiring.inputs = OmegaConf.create({})
        if not hasattr(step.wiring, "outputs"):
            step.wiring.outputs = OmegaConf.create({})

        step.wiring.inputs.model = f"fitted_{step.id}"
        step.wiring.inputs.features = "preprocessed_data"
        step.wiring.outputs.features = (
            step.wiring.outputs.features
            if hasattr(step.wiring.outputs, "features")
            else "cluster_features"
        )
        step.wiring.outputs.model = f"fitted_{step.id}"

    def _transform_preprocessor_in_pipeline(self, step: Any, alias: str) -> None:
        """Transform preprocessor step in sub-pipeline."""
        self._setup_step_for_load_mode(step, alias)
        self._remove_training_configs(step)
        if hasattr(step, "wiring") and "inputs" in step.wiring:
            delattr(step.wiring, "inputs")

    def _transform_clustering_in_pipeline(self, step: Any, _alias: str) -> None:
        """Transform clustering step in sub-pipeline."""
        self._setup_clustering_wiring(step)
        self._remove_training_configs(step)

    def _extract_base_name(self, step_id: str) -> str:
        """Extract base name from model producer step ID."""
        return step_id.replace("_features", "").replace("_model", "")

    def _is_clustering_model(self, mp: Any) -> bool:
        """Determine whether model producer is a clustering model."""
        if mp.type == "clustering":
            return True

        if mp.type == "dynamic_adapter" and hasattr(mp, "class_path"):
            class_path = mp.class_path.lower()
            return "cluster" in class_path or "kmeans" in class_path

        return False

    def _extract_features_input(self, mp: Any) -> str:
        """Extract evaluator features input from model producer wiring."""
        if not hasattr(mp, "wiring") or "inputs" not in mp.wiring:
            return "preprocessed_data"

        input_keys = mp.wiring.inputs

        if "X" in input_keys:
            return input_keys["X"]

        if "features" in input_keys:
            return input_keys["features"]

        return "preprocessed_data"

    def _collect_model_producer_ids(
        self,
        all_model_producers: Optional[List[Any]],
    ) -> set[str]:
        """Collect model producer step IDs."""
        if not all_model_producers:
            return set()

        return {p.id for p in all_model_producers}

    def _is_valid_evaluator_dependency(
        self,
        dep: str,
        model_producer_ids: set[str],
    ) -> bool:
        """Check whether dependency should be kept for evaluator."""
        if dep.startswith("tune_"):
            return False

        if dep in model_producer_ids:
            return False

        return True

    def _build_load_map(
        self,
        all_model_producers: List[Any],
        all_preprocessors: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Dict[str, str]]:
        """Build load_map for MLflow loader."""
        load_map = [
            {"step_id": mp.id, "context_key": f"fitted_{mp.id}"}
            for mp in all_model_producers
        ]

        for prep in all_preprocessors:
            load_map.append({"step_id": prep.id, "context_key": f"fitted_{prep.id}"})

        # Legacy compatibility
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            load_map.append(
                {"step_id": preprocessor_step.id, "context_key": "transform_manager"}
            )

        return load_map

    def _add_mlflow_loader(
        self,
        new_steps: List[Any],
        init_id: str,
        alias: str,
        load_map: List[Dict[str, str]],
    ) -> None:
        """Add MLflow loader step."""
        new_steps.append(
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
