"""Evaluation pipeline builder."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from .dependency_builder import DependencyBuilder
from .loader_builder import LoaderBuilder
from .step_analyzer import StepAnalyzer, StepExtractor
from .step_transformer import StepTransformer


class EvalBuilder:
    """Builds evaluation pipeline from training pipeline."""

    def __init__(self, train_steps: List[Any]) -> None:
        """Initialize eval pipeline builder.

        Args:
            train_steps: Original training pipeline steps.
        """
        self.train_steps = train_steps
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
    ) -> List[Any]:
        """Build complete evaluation pipeline.

        Args:
            alias: Model alias for loading.
            init_id: Initialization step ID.
            preprocessor_step: Legacy preprocessor step if exists.

        Returns:
            List of eval pipeline steps.
        """
        steps: List[Any] = []

        data_loader = next(
            (s for s in self.train_steps if s.type == "data_loader"), None
        )
        if data_loader:
            steps.append(data_loader)

        preprocessors = self.extractor.extract_preprocessors(self.train_steps)
        producers = self.extractor.extract_model_producers(self.train_steps)

        self.loader_builder.add_mlflow_loader(
            steps, init_id, alias, preprocessors, producers, preprocessor_step
        )

        self._add_legacy_preprocessor(
            steps, preprocessor_step, init_id, preprocessors, data_loader
        )

        self._add_preprocessors(steps, alias, init_id, data_loader)

        special, special_ids, branch_ids = self._process_special_steps(alias)
        steps.extend(special)

        evaluator_ids = self._add_evaluators(
            steps,
            producers,
            branch_ids,
            init_id,
            next(
                (s.id for s in self.train_steps if s.type == "preprocessor"),
                None,
            ),
            [
                sid
                for sid, step in zip(special_ids, special)
                if hasattr(step, "pipeline")
            ],
        )

        evaluator_ids.extend(
            [
                sid
                for sid, step in zip(special_ids, special)
                if hasattr(step, "condition")
            ]
        )

        self._add_auxiliary_steps(steps, evaluator_ids)

        return steps

    def _add_legacy_preprocessor(
        self,
        steps: List[Any],
        preprocessor_step: Optional[Any],
        init_id: str,
        all_preprocessors: List[Any],
        data_loader: Optional[Any],
    ) -> None:
        """Add legacy top-level preprocessor if needed."""
        if not preprocessor_step:
            return

        if any(p.id == preprocessor_step.id for p in all_preprocessors):
            return

        prep = copy.deepcopy(preprocessor_step)
        prep.is_train = False
        prep.alias = init_id
        prep.instance_key = "transform_manager"
        prep.depends_on = [init_id]

        if data_loader:
            prep.depends_on.append("load_data")

        steps.append(prep)

    def _add_preprocessors(
        self,
        steps: List[Any],
        alias: str,
        init_id: str,
        data_loader: Optional[Any],
    ) -> None:
        """Add preprocessors and adapters to pipeline."""
        for step in self.train_steps:
            if step.type == "preprocessor":
                prep = copy.deepcopy(step)
                prep.is_train = False
                prep.alias = alias
                prep.instance_key = f"fitted_{step.id}"
                prep.depends_on = [init_id]

                if data_loader:
                    prep.depends_on.append("load_data")

                steps.append(prep)

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

    def _should_add_adapter(self, step: Any) -> bool:
        """Check if dynamic adapter should be added."""
        if not hasattr(step, "log_artifact") or not step.log_artifact:
            return False

        if not hasattr(step, "artifact_type"):
            return False

        return step.artifact_type == "preprocess"

    def _process_special_steps(
        self, alias: str
    ) -> Tuple[List[Any], List[str], Set[str]]:
        """Process sub-pipelines and branches."""
        special = []
        special_ids = []
        branch_ids: Set[str] = set()

        for step in self.train_steps:
            if step.type == "sub_pipeline":
                transformed = self._transform_sub_pipeline(step, alias)
                special.append(transformed)
                special_ids.append(step.id)

            elif step.type == "branch":
                transformed = self._transform_branch(step)
                special.append(transformed)
                special_ids.append(step.id)

                for branch_name in ["if_true", "if_false"]:
                    if not hasattr(step, branch_name):
                        continue

                    branch = getattr(step, branch_name)
                    if hasattr(branch, "id"):
                        branch_ids.add(branch.id)

            elif step.type == "parallel":
                special_ids.append(step.id)

        return special, special_ids, branch_ids

    def _transform_sub_pipeline(self, step: Any, alias: str) -> Any:
        """Transform sub-pipeline for eval mode."""
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
                self.transformer.transform_model_step(sub_step)

        return transformed

    def _transform_branch(self, step: Any) -> Any:
        """Transform branch step for eval mode."""
        producer_ids = self.extractor.collect_producer_ids(
            self.extractor.extract_model_producers(self.train_steps)
        )

        resolved_deps = self.dependency_builder.resolve_dependencies(step, producer_ids)

        if hasattr(step, "depends_on"):
            step.depends_on = resolved_deps

        transformed = copy.deepcopy(step)

        if hasattr(transformed, "if_true"):
            evaluator = self._branch_to_evaluator(transformed.if_true)
            if evaluator:
                transformed.if_true = evaluator

        if hasattr(transformed, "if_false"):
            evaluator = self._branch_to_evaluator(transformed.if_false)
            if evaluator:
                transformed.if_false = evaluator

        return transformed

    def _branch_to_evaluator(self, branch: Any) -> Optional[DictConfig]:
        """Convert branch to evaluator step."""
        if not self.analyzer.is_model_producer(branch):
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

    def _create_evaluator_wiring(
        self, model_id: str, model_type: str, outputs: Optional[Any]
    ) -> Dict[str, Any]:
        """Create wiring for evaluator step."""
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

    def _add_evaluators(
        self,
        steps: List[Any],
        producers: List[Any],
        branch_ids: Set[str],
        init_id: str,
        preprocess_id: Optional[str],
        sub_pipeline_ids: List[str],
    ) -> List[str]:
        """Add evaluator steps to pipeline."""
        evaluator_ids = []

        for producer in producers:
            if producer.id in branch_ids:
                continue

            if self._is_clustering_adapter(producer):
                continue

            evaluator = self._make_evaluator(
                producer, init_id, preprocess_id, producers
            )

            for sid in sub_pipeline_ids:
                if sid not in evaluator.depends_on:
                    evaluator.depends_on.append(sid)

            evaluator_ids.append(evaluator.id)
            steps.append(evaluator)

        return evaluator_ids

    def _is_clustering_adapter(self, step: Any) -> bool:
        """Check if step is clustering dynamic adapter."""
        if step.type != "dynamic_adapter":
            return False

        if not hasattr(step, "class_path"):
            return False

        path_lower = step.class_path.lower()
        return "cluster" in path_lower or "kmeans" in path_lower

    def _make_evaluator(
        self,
        producer: Any,
        init_id: str,
        preprocessor_id: Optional[str],
        all_producers: List[Any],
    ) -> DictConfig:
        """Create evaluator config for model producer."""
        base_name = self.transformer.extract_base_name(producer.id)
        eval_id = f"{base_name}_evaluate"

        is_clustering = self.analyzer.is_clustering(producer)
        features_input = self.analyzer.extract_features_input(producer)

        inputs = self._build_evaluator_inputs(
            producer.id, features_input, is_clustering
        )

        producer_ids = self.extractor.collect_producer_ids(all_producers)

        deps = self.dependency_builder.build_eval_dependencies(
            producer, init_id, preprocessor_id, producer_ids
        )

        config: Dict[str, Any] = {
            "id": eval_id,
            "type": "evaluator",
            "enabled": True,
            "depends_on": deps,
            "wiring": {
                "inputs": inputs,
                "outputs": {"metrics": f"{base_name}_metrics"},
            },
        }

        if is_clustering:
            config["step_eval_type"] = "clustering"

        return OmegaConf.create(config)

    def _build_evaluator_inputs(
        self, mp_id: str, features: str, is_clustering: bool
    ) -> Dict[str, Any]:
        """Build evaluator input wiring."""
        inputs: Dict[str, Any] = {
            "features": features,
            "model": f"fitted_{mp_id}",
        }

        if not is_clustering:
            inputs["targets"] = "target_data"

        return inputs

    def _add_auxiliary_steps(self, steps: List[Any], evaluator_ids: List[str]) -> None:
        """Add logger and profiling steps."""
        for step in self.train_steps:
            if step.type not in ["logger", "profiling"]:
                continue

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

            steps.append(aux)
