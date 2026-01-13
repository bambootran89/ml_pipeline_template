from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from .base_transform_mixin import BaseTransformMixin


class ServePipelineMixin(BaseTransformMixin):
    """Mixin providing pipeline transformation logic for inference serving.

    This mixin encapsulates all utilities required to convert a training
    pipeline configuration into a serving (inference-only) pipeline,
    without altering any model logic or execution semantics.

    Core responsibilities:
    - Switch all applicable steps from training mode to inference/load mode
      (`is_train=False`, alias assignment, instance_key setup).
    - Remove training-only dependencies (e.g. `load_data`) that are not
      applicable in serving environments.
    - Rewrite wiring for inference steps to consume fitted models and
      preprocessed features.
    - Transform sub-pipelines so they operate in inference mode only.
    - Transform branch steps (`if_true` / `if_false`) into inference steps
      where the branch produces a model.
    - Handle complex pipeline structures (sub-pipeline, branch, parallel)
      and ensure correct dependency ordering.
    - Generate inference steps for all eligible model producers.
    - Preserve backward compatibility for legacy preprocessor placement.
    - Append final profiling for post-inference inspection.

    The mixin assumes the host class provides shared helpers such as:
    `_is_model_producer`, `_remove_training_configs`,
    `_extract_model_producers_recursive`, `_extract_preprocessors_recursive`,
    `_add_mlflow_loader`, and `_build_load_map`.

    This class is purely structural: it reorganizes pipeline steps and
    wiring for serving, while keeping model behavior and data flow intact.
    """

    def _transform_sub_pipeline_for_serve(
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for serve mode."""
        transformed = copy.deepcopy(sub_pipeline_step)
        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        if hasattr(transformed, "depends_on"):
            transformed.depends_on = [
                dep for dep in transformed.depends_on if dep != "load_data"
            ]
            if not transformed.depends_on:
                delattr(transformed, "depends_on")

        for step in transformed.pipeline.steps:
            if step.type == "preprocessor":
                self._transform_preprocessor_in_pipeline(step, alias)
            elif step.type == "clustering":
                self._transform_clustering_in_pipeline(step, alias)

        return transformed

    def _create_inference_wiring(self, model_id: str) -> Dict[str, Any]:
        """Create wiring configuration for inference."""
        return {
            "inputs": {
                "model": f"fitted_{model_id}",
                "features": "preprocessed_data",
            },
            "outputs": {
                "predictions": f"{model_id}_predictions",
            },
        }

    def _transform_branch_to_inference(self, branch: Any) -> Optional[DictConfig]:
        """Transform a branch (if_true/if_false) to inference."""
        if not self._is_model_producer(branch):
            return None

        inf_id = f"{branch.id}_inference"
        wiring = self._create_inference_wiring(branch.id)

        config = {
            "id": inf_id,
            "type": "inference",
            "enabled": True,
            "depends_on": ["preprocess"],
            "output_as_feature": getattr(branch, "output_as_feature", False),
            "wiring": wiring,
        }

        return OmegaConf.create(config)

    def _transform_branch_step_for_serve(self, branch_step: Any) -> Any:
        """Transform a branch step for serve mode."""
        transformed = copy.deepcopy(branch_step)

        if hasattr(transformed, "if_true"):
            inference = self._transform_branch_to_inference(transformed.if_true)
            if inference:
                transformed.if_true = inference

        if hasattr(transformed, "if_false"):
            inference = self._transform_branch_to_inference(transformed.if_false)
            if inference:
                transformed.if_false = inference

        return transformed

    def _add_preprocessors_to_serve(
        self,
        new_steps: List[Any],
        train_steps: List[Any],
        alias: str,
        init_id: str,
    ) -> str:
        """Add preprocessors and dynamic adapters to serve pipeline."""
        last_id = init_id
        for step in train_steps:
            if step.type == "preprocessor" and any(
                s.id == step.id for s in train_steps
            ):
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"fitted_{step.id}"
                prep.depends_on = [init_id]
                new_steps.append(prep)
                last_id = prep.id
            elif step.type == "dynamic_adapter":
                if (
                    hasattr(step, "log_artifact")
                    and step.log_artifact
                    and hasattr(step, "artifact_type")
                    and step.artifact_type == "preprocess"
                ):
                    adapter = copy.deepcopy(step)
                    adapter.is_train = False
                    adapter.instance_key = f"fitted_{step.id}"
                    adapter.depends_on = [init_id]
                    if hasattr(step, "depends_on"):
                        adapter.depends_on.extend(step.depends_on)
                    self._remove_training_configs(adapter)
                    new_steps.append(adapter)
                    last_id = adapter.id
        return last_id

    def _extract_branch_producer_ids(self, step: Any) -> Set[str]:
        """Extract producer IDs from branch step."""
        producer_ids: Set[str] = set()
        for branch_name in ["if_true", "if_false"]:
            if hasattr(step, branch_name):
                branch = getattr(step, branch_name)
                if hasattr(branch, "id"):
                    producer_ids.add(branch.id)
        return producer_ids

    def _handle_special_steps_serve(
        self,
        new_steps: List[Any],
        train_steps: List[Any],
        alias: str,
        has_sub_pipeline: bool,
        has_branch: bool,
    ) -> Tuple[str, Set[str]]:
        """Handle sub-pipelines and branches in serve mode."""
        last_id = "init_artifacts"
        branch_producer_ids: Set[str] = set()

        if has_sub_pipeline:
            for step in train_steps:
                if step.type == "sub_pipeline":
                    transformed = self._transform_sub_pipeline_for_serve(step, alias)
                    new_steps.append(transformed)
                    last_id = step.id

        if has_branch:
            for step in train_steps:
                if step.type == "branch":
                    transformed = self._transform_branch_step_for_serve(step)
                    new_steps.append(transformed)
                    last_id = step.id
                    branch_producer_ids.update(self._extract_branch_producer_ids(step))

        return last_id, branch_producer_ids

    def _get_inference_producers(
        self,
        all_model_producers: List[Any],
        train_steps: List[Any],
        branch_producer_ids: Set[str],
        has_sub_pipeline: bool,
        has_branch: bool,
        has_parallel: bool,
        model_producers: List[Any],
    ) -> List[Any]:
        """Determine which model producers need inference steps."""
        if has_sub_pipeline or has_branch or has_parallel:
            if has_parallel:
                return [
                    mp for mp in all_model_producers if mp.id not in branch_producer_ids
                ]
            else:
                return [
                    mp
                    for mp in all_model_producers
                    if mp.id not in branch_producer_ids
                    and any(s.id == mp.id for s in train_steps)
                ]
        else:
            return model_producers

    def _add_inference_steps(
        self,
        new_steps: List[Any],
        producers: List[Any],
        init_id: str,
        last_id: str,
    ) -> str:
        """Add inference steps to serve pipeline."""
        for mp in producers:
            inf_id = f"{mp.id}_inference"
            inf_step = OmegaConf.create(
                {
                    "id": inf_id,
                    "type": "inference",
                    "enabled": True,
                    "depends_on": [init_id, last_id],
                    "output_as_feature": getattr(mp, "output_as_feature", False),
                    "wiring": {
                        "inputs": {
                            "model": f"fitted_{mp.id}",
                            "features": "preprocessed_data",
                        },
                        "outputs": {"predictions": f"{mp.id}_predictions"},
                    },
                }
            )
            new_steps.append(inf_step)
            last_id = inf_id
        return last_id

    def build_serve_steps(
        self,
        alias: str,
        init_id: str,
        model_producers: List[Any],
        preprocessor_step: Optional[Any],
        train_steps: List[Any],
    ) -> List[Any]:
        """Build steps for serve pipeline."""
        new_steps: List[Any] = []

        all_preprocessors = self._extract_preprocessors_recursive(train_steps)
        all_model_producers = self._extract_model_producers_recursive(train_steps)

        self._add_mlflow_loader(
            new_steps,
            init_id,
            alias,
            self._build_load_map(
                all_model_producers, all_preprocessors, preprocessor_step
            ),
        )

        last_id = self._add_preprocessors_to_serve(
            new_steps, train_steps, alias, init_id
        )

        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            legacy_prep = copy.deepcopy(preprocessor_step)
            legacy_prep.is_train = False
            legacy_prep.alias = alias
            legacy_prep.instance_key = "transform_manager"
            legacy_prep.depends_on = [init_id]
            new_steps.append(legacy_prep)
            last_id = legacy_prep.id

        pipeline_types = {
            "has_sub": any(s.type == "sub_pipeline" for s in train_steps),
            "has_branch": any(s.type == "branch" for s in train_steps),
            "has_parallel": any(s.type == "parallel" for s in train_steps),
        }

        special_last_id, branch_ids = self._handle_special_steps_serve(
            new_steps,
            train_steps,
            alias,
            pipeline_types["has_sub"],
            pipeline_types["has_branch"],
        )
        if pipeline_types["has_sub"] or pipeline_types["has_branch"]:
            last_id = special_last_id

        last_id = self._add_inference_steps(
            new_steps,
            self._get_inference_producers(
                all_model_producers,
                train_steps,
                branch_ids,
                pipeline_types["has_sub"],
                pipeline_types["has_branch"],
                pipeline_types["has_parallel"],
                model_producers,
            ),
            init_id,
            last_id,
        )

        new_steps.append(
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

        return new_steps
