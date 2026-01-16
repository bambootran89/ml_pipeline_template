"""Serving pipeline builder."""
# pylint: disable=R0911

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from .dependency_builder import DependencyBuilder
from .loader_builder import LoaderBuilder
from .step_analyzer import StepAnalyzer, StepExtractor
from .step_transformer import StepTransformer


class ServeBuilder:
    """Builds serving pipeline from training pipeline."""

    def __init__(
        self, train_steps: List[Any], experiment_type: str = "tabular"
    ) -> None:
        """Initialize serve pipeline builder.

        Args:
            train_steps: Original training pipeline steps.
            experiment_type: Type of experiment (timeseries, tabular).
        """
        self.train_steps = train_steps
        self.experiment_type = experiment_type
        self.analyzer = StepAnalyzer()
        self.extractor = StepExtractor()
        self.dependency_builder = DependencyBuilder(train_steps)
        self.loader_builder = LoaderBuilder()
        self.transformer = StepTransformer()

    def build(
        self,
        alias: str,
        init_id: str,
        preprocessor_step: Optional[Any],
        feature_pipeline: Optional[Any] = None,
    ) -> List[Any]:
        """Build complete serving pipeline.

        Args:
            alias: Model alias for loading.
            init_id: Initialization step ID.
            preprocessor_step: Legacy preprocessor step if exists.
            feature_pipeline: Feature pipeline config.

        Returns:
            List of serve pipeline steps.
        """
        steps: List[Any] = []

        preprocessors = self.extractor.extract_preprocessors(self.train_steps)
        producers = self.extractor.extract_model_producers(self.train_steps)

        self.loader_builder.add_mlflow_loader(
            steps,
            init_id,
            alias,
            preprocessors,
            producers,
            preprocessor_step,
            feature_pipeline,
        )

        last_id = self._add_preprocessors(steps, alias, init_id)

        last_id = self._add_legacy_preprocessor(
            steps, preprocessor_step, alias, init_id, preprocessors, last_id
        )

        pipeline_flags = {
            "has_sub": any(s.type == "sub_pipeline" for s in self.train_steps),
            "has_branch": any(s.type == "branch" for s in self.train_steps),
            "has_parallel": any(s.type == "parallel" for s in self.train_steps),
        }

        special_last_id, branch_ids = self._handle_special_steps(
            steps, alias, pipeline_flags["has_sub"], pipeline_flags["has_branch"]
        )

        if pipeline_flags["has_sub"] or pipeline_flags["has_branch"]:
            last_id = special_last_id

        inference_producers = self._get_inference_producers(
            producers, branch_ids, pipeline_flags
        )

        last_id = self._add_inference_steps(
            steps, inference_producers, init_id, last_id
        )

        self._add_final_profiling(steps, last_id)

        return steps

    def _add_preprocessors(self, steps: List[Any], alias: str, init_id: str) -> str:
        """Add preprocessors and adapters to pipeline."""
        last_id = init_id

        for step in self.train_steps:
            if step.type == "preprocessor":
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"fitted_{step.id}"
                prep.depends_on = [init_id]
                steps.append(prep)
                last_id = prep.id

            elif step.type == "dynamic_adapter":
                if not self._should_add_adapter(step):
                    continue

                adapter = copy.deepcopy(step)
                adapter.is_train = False
                adapter.instance_key = f"fitted_{step.id}"
                adapter.depends_on = [init_id]

                if hasattr(step, "depends_on"):
                    adapter.depends_on.extend(step.depends_on)

                self.transformer.remove_training_configs(adapter)
                steps.append(adapter)
                last_id = adapter.id

        return last_id

    def _should_add_adapter(self, step: Any) -> bool:
        """Check if dynamic adapter should be added."""
        if not hasattr(step, "log_artifact") or not step.log_artifact:
            return False

        if not hasattr(step, "artifact_type"):
            return False

        return step.artifact_type == "preprocess"

    def _add_legacy_preprocessor(
        self,
        steps: List[Any],
        preprocessor_step: Optional[Any],
        alias: str,
        init_id: str,
        all_preprocessors: List[Any],
        current_last_id: str,
    ) -> str:
        """Add legacy preprocessor if needed."""
        if not preprocessor_step:
            return current_last_id

        if any(p.id == preprocessor_step.id for p in all_preprocessors):
            return current_last_id

        prep = copy.deepcopy(preprocessor_step)
        prep.is_train = False
        prep.alias = alias
        prep.instance_key = "transform_manager"
        prep.depends_on = [init_id]
        steps.append(prep)

        return prep.id

    def _handle_special_steps(
        self,
        steps: List[Any],
        alias: str,
        has_sub: bool,
        has_branch: bool,
    ) -> Tuple[str, Set[str]]:
        """Handle sub-pipelines and branches."""
        last_id = "init_artifacts"
        branch_ids: Set[str] = set()

        if has_sub:
            for step in self.train_steps:
                if step.type == "sub_pipeline":
                    transformed = self._transform_sub_pipeline(step, alias)
                    steps.append(transformed)
                    last_id = step.id

        if has_branch:
            for step in self.train_steps:
                if step.type == "branch":
                    transformed = self._transform_branch(step)
                    steps.append(transformed)
                    last_id = step.id
                    branch_ids.update(self._extract_branch_ids(step))

        return last_id, branch_ids

    def _transform_sub_pipeline(self, step: Any, alias: str) -> Any:
        """Transform sub-pipeline for serve mode."""
        producer_ids = self.extractor.collect_producer_ids(
            self.extractor.extract_model_producers(self.train_steps)
        )

        resolved_deps = self.dependency_builder.resolve_dependencies(step, producer_ids)

        if hasattr(step, "depends_on"):
            step.depends_on = resolved_deps

        transformed = copy.deepcopy(step)

        if not hasattr(transformed, "pipeline"):
            return transformed

        if not hasattr(transformed.pipeline, "steps"):
            return transformed

        for sub_step in transformed.pipeline.steps:
            if sub_step.type == "preprocessor":
                self.transformer.transform_preprocessor(sub_step, alias)

            elif sub_step.type in ["clustering", "model"]:
                sub_step.type = "inference"
                self.transformer.transform_model_step(sub_step)

                sub_step.wiring.outputs.predictions = (
                    sub_step.wiring.outputs.predictions
                    if hasattr(sub_step.wiring.outputs, "predictions")
                    else f"{sub_step['id']}_predictions"
                )

        return transformed

    def _transform_branch(self, step: Any) -> Any:
        """Transform branch step for serve mode."""
        transformed = copy.deepcopy(step)

        if hasattr(transformed, "if_true"):
            inference = self._branch_to_inference(transformed.if_true)
            if inference:
                transformed.if_true = inference

        if hasattr(transformed, "if_false"):
            inference = self._branch_to_inference(transformed.if_false)
            if inference:
                transformed.if_false = inference

        return transformed

    def _resolve_base_features_key(self, step: Any) -> str:
        """Resolve base features key from step wiring."""
        if hasattr(step, "wiring") and hasattr(step.wiring, "inputs"):
            inputs = step.wiring.inputs
            # Prioritize standard keys
            for key in ["X", "features", "data", "input"]:
                if key in inputs:
                    return inputs[key]
        return "preprocessed_data"

    def _branch_to_inference(self, branch: Any) -> Optional[DictConfig]:
        """Convert branch to inference step."""
        if not self.analyzer.is_model_producer(branch):
            return None

        inf_id = f"{branch.id}_inference"

        config = {
            "id": inf_id,
            "type": "feature_inference",
            "enabled": True,
            "depends_on": ["preprocess"],
            "source_model_key": f"fitted_{branch.id}",
            "base_features_key": self._resolve_base_features_key(branch),
            "output_key": f"{branch.id}_predictions",
            "apply_windowing": self._should_apply_windowing(branch.id),
        }

        return OmegaConf.create(config)

    def _create_inference_wiring(self, model_id: str) -> Dict[str, Any]:
        """Create wiring for inference step."""
        return {
            "inputs": {
                "model": f"fitted_{model_id}",
                "features": "preprocessed_data",
            },
            "outputs": {
                "predictions": f"{model_id}_predictions",
            },
        }

    def _extract_branch_ids(self, step: Any) -> Set[str]:
        """Extract producer IDs from branch step."""
        ids: Set[str] = set()

        for branch_name in ["if_true", "if_false"]:
            if not hasattr(step, branch_name):
                continue

            branch = getattr(step, branch_name)
            if hasattr(branch, "id"):
                ids.add(branch.id)

        return ids

    def _get_inference_producers(
        self,
        all_producers: List[Any],
        branch_ids: Set[str],
        pipeline_flags: Dict[str, bool],
    ) -> List[Any]:
        """Determine which producers need inference steps."""
        has_sub = pipeline_flags.get("has_sub", False)
        has_branch = pipeline_flags.get("has_branch", False)
        has_parallel = pipeline_flags.get("has_parallel", False)

        if not (has_sub or has_branch or has_parallel):
            return all_producers

        if has_parallel:
            return [p for p in all_producers if p.id not in branch_ids]

        return [
            p
            for p in all_producers
            if p.id not in branch_ids and any(s.id == p.id for s in self.train_steps)
        ]

    def _add_inference_steps(
        self,
        steps: List[Any],
        producers: List[Any],
        init_id: str,
        last_id: str,
    ) -> str:
        """Add inference steps to pipeline."""
        for producer in producers:
            inf_id = f"{producer.id}_inference"

            inf_step = OmegaConf.create(
                {
                    "id": inf_id,
                    "type": "feature_inference",
                    "enabled": True,
                    "depends_on": [init_id, last_id],
                    "source_model_key": f"fitted_{producer.id}",
                    "base_features_key": self._resolve_base_features_key(producer),
                    "output_key": f"{producer.id}_predictions",
                    "apply_windowing": self._should_apply_windowing(producer.id),
                }
            )

            steps.append(inf_step)
            last_id = inf_id

        return last_id

    def _should_apply_windowing(self, step_id: str) -> bool:
        """Check if windowing should be applied based on step ID."""
        if self.experiment_type != "timeseries":
            return False

        # Find the original training step to check its configuration
        train_step = self.extractor.find_step_by_id(self.train_steps, step_id)
        if not train_step:
            return True

        step_type = train_step.get("type", "")

        # Standard trainers and clustering steps in timeseries expect windowed data
        if step_type in ["trainer", "clustering", "framework_model"]:
            return True

        # For dynamic adapters, check if they received a datamodule in training
        if step_type == "dynamic_adapter":
            if "wiring" in train_step and "inputs" in train_step.wiring:
                inputs = train_step.wiring.inputs
                if "datamodule" in inputs:
                    return True
            return False

        # Fallback for other steps in timeseries
        step_lower = step_id.lower()
        if any(x in step_lower for x in ["impute", "pca", "scaler"]):
            return False

        return True

    def _add_final_profiling(self, steps: List[Any], last_id: str) -> None:
        """Add final profiling step."""
        steps.append(
            OmegaConf.create(
                {
                    "id": "final_profiling",
                    "type": "profiling",
                    "enabled": True,
                    "depends_on": [last_id],
                    "exclude_keys": [
                        "cfg",
                        "preprocessor",
                        "df",
                        "train_df",
                        "val_df",
                        "test_df",
                    ],
                }
            )
        )
