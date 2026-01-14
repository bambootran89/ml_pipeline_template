from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from .base_mixin import BaseTransformMixin


class EvalPipelineMixin(BaseTransformMixin):
    """Mixin providing evaluation-specific pipeline transformation utilities.

    This mixin encapsulates all logic required to transform a training pipeline
    into an evaluation (eval) pipeline without altering the original training
    behavior or artifacts.
    """

    def _transform_sub_pipeline_for_eval(
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for eval mode."""
        transformed = copy.deepcopy(sub_pipeline_step)
        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        model_producer_ids = self._collect_model_producer_ids(
            self._extract_model_producers_recursive(self.train_steps),
        )
        depends_on = self._corect_dependencies(
            mp=sub_pipeline_step, model_producer_ids=model_producer_ids
        )
        if hasattr(sub_pipeline_step, "depends_on"):
            sub_pipeline_step.depends_on = depends_on
        transformed = copy.deepcopy(sub_pipeline_step)
        for step in transformed.pipeline.steps:
            if step.type == "preprocessor":
                self._transform_preprocessor_in_pipeline(step, alias)
            elif step.type in ["clustering", "model"]:
                self._transform_model_or_clustering_in_pipeline(step, alias)

        return transformed

    def _create_evaluator_wiring(
        self, model_id: str, model_type: str, outputs: Optional[Any]
    ) -> Dict[str, Any]:
        """Create wiring configuration for evaluator."""
        wiring = {
            "inputs": {
                "model": f"fitted_{model_id}",
                "features": "preprocessed_data",
            },
            "outputs": {
                "metrics": (
                    outputs.get("metrics", "evaluation_metrics")
                    if outputs
                    else "evaluation_metrics"
                )
            },
        }
        if model_type != "clustering":
            wiring["inputs"]["targets"] = "target_data"
        return wiring

    def _transform_branch_to_evaluator(self, branch: Any) -> Optional[DictConfig]:
        """Transform a branch (if_true/if_false) to evaluator."""
        if not self._is_model_producer(branch):
            return None

        eval_id = (
            f"{branch.id}_evaluate"
            if not branch.id.endswith("_evaluate")
            else branch.id
        )

        outputs = (
            branch.wiring.outputs
            if hasattr(branch, "wiring") and hasattr(branch.wiring, "outputs")
            else None
        )

        wiring = self._create_evaluator_wiring(branch.id, branch.type, outputs)

        config = {
            "id": eval_id,
            "type": "evaluator",
            "enabled": True,
            "depends_on": ["preprocess"],
            "wiring": wiring,
        }

        if branch.type == "clustering":
            config["step_eval_type"] = "clustering"

        return OmegaConf.create(config)

    def _transform_branch_step_for_eval(self, branch_step: Any) -> Any:
        """Transform a branch step for eval mode."""

        model_producer_ids = self._collect_model_producer_ids(
            self._extract_model_producers_recursive(self.train_steps),
        )
        depends_on = self._corect_dependencies(
            mp=branch_step, model_producer_ids=model_producer_ids
        )
        if hasattr(branch_step, "depends_on"):
            branch_step.depends_on = depends_on
        transformed = copy.deepcopy(branch_step)
        if hasattr(transformed, "if_true"):
            evaluator = self._transform_branch_to_evaluator(transformed.if_true)
            if evaluator:
                transformed.if_true = evaluator

        if hasattr(transformed, "if_false"):
            evaluator = self._transform_branch_to_evaluator(transformed.if_false)
            if evaluator:
                transformed.if_false = evaluator
        return transformed

    def _make_evaluator_config(
        self,
        mp: Any,
        init_id: str,
        preprocessor_id: Optional[str],
        all_model_producers: Optional[List[Any]] = None,
    ) -> DictConfig:
        """Create evaluator config for a model producer step."""
        base_name = self._extract_base_name(mp.id)
        eval_id = f"{base_name}_evaluate"

        is_clustering = self._is_clustering_model(mp)
        features_input = self._extract_features_input(mp)

        inputs = self._build_evaluator_inputs(
            mp_id=mp.id,
            features_input=features_input,
            is_clustering=is_clustering,
        )

        depends_on = self._build_evaluator_dependencies(
            mp=mp,
            init_id=init_id,
            preprocessor_id=preprocessor_id,
            all_model_producers=all_model_producers,
        )
        step_cfg: Dict[str, Any] = {
            "id": eval_id,
            "type": "evaluator",
            "enabled": True,
            "depends_on": depends_on,
            "wiring": {
                "inputs": inputs,
                "outputs": {"metrics": f"{base_name}_metrics"},
            },
        }

        if is_clustering:
            step_cfg["step_eval_type"] = "clustering"

        return OmegaConf.create(step_cfg)

    def _build_evaluator_inputs(
        self,
        mp_id: str,
        features_input: str,
        is_clustering: bool,
    ) -> Dict[str, Any]:
        """Build evaluator input wiring."""
        inputs: Dict[str, Any] = {
            "features": features_input,
            "model": f"fitted_{mp_id}",
        }

        if not is_clustering:
            inputs["targets"] = "target_data"

        return inputs

    def _build_evaluator_dependencies(
        self,
        mp: Any,
        init_id: str,
        preprocessor_id: Optional[str],
        all_model_producers: Optional[List[Any]],
    ) -> List[str]:
        """Build evaluator step dependencies."""
        model_producer_ids = self._collect_model_producer_ids(
            all_model_producers,
        )
        depends_on = [init_id]

        if hasattr(mp, "depends_on") and mp.depends_on:
            for dep in mp.depends_on:
                if self._is_valid_evaluator_dependency(dep, model_producer_ids):
                    # Check if dep is inside a sub-pipeline
                    if self.train_steps:
                        parent_pipeline = self._find_parent_sub_pipeline(
                            self.train_steps, dep
                        )
                        if parent_pipeline and parent_pipeline not in depends_on:
                            # Replace internal step with parent sub-pipeline
                            depends_on.append(parent_pipeline)
                        elif not parent_pipeline and dep not in depends_on:
                            depends_on.append(dep)
                    else:
                        if dep not in depends_on:
                            depends_on.append(dep)

        if preprocessor_id and preprocessor_id not in depends_on:
            depends_on.append(preprocessor_id)
        return depends_on

    def _add_top_level_preprocessor_eval(
        self,
        new_steps: List[Any],
        preprocessor_step: Any,
        init_id: str,
        all_preprocessors: List[Any],
        data_loader: Optional[Any],
    ) -> None:
        """Add top-level preprocessor for eval mode (legacy)."""
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            prep = copy.deepcopy(preprocessor_step)
            prep.is_train = False
            prep.alias = init_id  # Using init_id as alias placeholder
            prep.instance_key = "transform_manager"
            prep.depends_on = [init_id]
            if data_loader:
                prep.depends_on.append("load_data")
            new_steps.append(prep)

    def _process_special_steps(
        self, train_steps: List[Any], alias: str
    ) -> Tuple[List[Any], List[str], Set[str]]:
        """Process sub-pipelines, branches, and parallel steps."""
        special_steps = []
        special_ids = []
        branch_producer_ids: Set[str] = set()

        for step in train_steps:
            if step.type == "sub_pipeline":
                transformed = self._transform_sub_pipeline_for_eval(step, alias)
                special_steps.append(transformed)
                special_ids.append(step.id)
            elif step.type == "branch":
                transformed = self._transform_branch_step_for_eval(step)
                special_steps.append(transformed)
                special_ids.append(step.id)
                for branch_name in ["if_true", "if_false"]:
                    if hasattr(step, branch_name):
                        branch = getattr(step, branch_name)
                        if hasattr(branch, "id"):
                            branch_producer_ids.add(branch.id)
            elif step.type == "parallel":
                special_ids.append(step.id)

        return special_steps, special_ids, branch_producer_ids

    def _add_preprocessors_to_eval(
        self,
        new_steps: List[Any],
        train_steps: List[Any],
        alias: str,
        init_id: str,
        data_loader: Optional[Any],
    ) -> None:
        """Add preprocessors and dynamic adapters to eval pipeline."""
        for step in train_steps:
            if step.type == "preprocessor" and any(
                s.id == step.id for s in train_steps
            ):
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"fitted_{step.id}"
                prep.depends_on = [init_id]
                if data_loader:
                    prep.depends_on.append("load_data")
                new_steps.append(prep)
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

    def _add_evaluators_to_eval(
        self,
        new_steps: List[Any],
        all_model_producers: List[Any],
        branch_producer_ids: Set[str],
        init_id: str,
        preprocess_id: Optional[str],
        sub_pipeline_ids: List[str],
    ) -> List[str]:
        """Add evaluators to eval pipeline. Returns list of evaluator IDs."""
        evaluator_ids = []
        for mp in all_model_producers:
            if mp.id not in branch_producer_ids:
                is_clustering_adapter = False
                if mp.type == "dynamic_adapter" and hasattr(mp, "class_path"):
                    class_path_lower = mp.class_path.lower()
                    is_clustering_adapter = (
                        "cluster" in class_path_lower or "kmeans" in class_path_lower
                    )

                if is_clustering_adapter:
                    continue

                ev = self._make_evaluator_config(
                    mp, init_id, preprocess_id, all_model_producers
                )
                if sub_pipeline_ids:
                    for sid in sub_pipeline_ids:
                        if sid not in ev.depends_on:
                            ev.depends_on.append(sid)
                evaluator_ids.append(ev.id)
                new_steps.append(ev)
        return evaluator_ids

    def _add_auxiliary_steps(
        self, new_steps: List[Any], train_steps: List[Any], evaluator_ids: List[str]
    ) -> None:
        """Add logger and profiling steps to eval pipeline."""
        for step in train_steps:
            if step.type in ["logger", "profiling"]:
                aux = copy.deepcopy(step)
                aux.depends_on = evaluator_ids
                if aux.type == "profiling":
                    aux.exclude_keys = [
                        "cfg",
                        "preprocessor",
                        "df",
                        "train_df",
                        "val_df",
                        "test_df",
                    ]
                new_steps.append(aux)

    def build_eval_steps(
        self,
        train_steps: List[Any],
        alias: str,
        init_id: str,
        _unused_model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for eval pipeline."""
        new_steps: List[Any] = []

        data_loader = next((s for s in train_steps if s.type == "data_loader"), None)
        if data_loader:
            new_steps.append(data_loader)

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

        self._add_top_level_preprocessor_eval(
            new_steps, preprocessor_step, init_id, all_preprocessors, data_loader
        )

        self._add_preprocessors_to_eval(
            new_steps, train_steps, alias, init_id, data_loader
        )

        special_steps, special_ids, branch_producer_ids = self._process_special_steps(
            train_steps, alias
        )
        new_steps.extend(special_steps)

        evaluator_ids = self._add_evaluators_to_eval(
            new_steps,
            all_model_producers,
            branch_producer_ids,
            init_id,
            next((s.id for s in train_steps if s.type == "preprocessor"), None),
            [
                sid
                for sid, step in zip(special_ids, special_steps)
                if hasattr(step, "pipeline")
            ],
        )

        evaluator_ids.extend(
            [
                sid
                for sid, step in zip(special_ids, special_steps)
                if hasattr(step, "condition")
            ]
        )
        self._add_auxiliary_steps(new_steps, train_steps, evaluator_ids)

        return new_steps
