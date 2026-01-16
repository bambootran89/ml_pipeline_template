"""Eval builder with feature pipeline support."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf

from .dependency_builder import DependencyBuilder
from .feature_pipeline_parser import EngineeredFeature, FeaturePipeline
from .loader_builder import LoaderBuilder
from .step_analyzer import StepAnalyzer, StepExtractor
from .step_transformer import StepTransformer


class EvalBuilder:
    """Evaluation pipeline builder."""

    # pylint: disable=R0911
    def __init__(
        self, train_steps: List[Any], experiment_type: str = "tabular"
    ) -> None:
        """Initialize eval builder.

        Parameters
        ----------
        train_steps : List[Any]
            Original training steps.
        experiment_type : str
            Type of experiment (timeseries, tabular).
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
        feature_pipeline: Optional[FeaturePipeline] = None,
    ) -> List[Any]:
        """Build evaluation pipeline.

        Parameters
        ----------
        alias : str
            Model alias for loading.
        init_id : str
            Init step ID.
        preprocessor_step : Optional[Any]
            Legacy preprocessor.
        feature_pipeline : Optional[FeaturePipeline]
            Feature pipeline config.

        Returns
        -------
        List[Any]
            Eval pipeline steps.
        """
        steps: List[Any] = []

        # Data loader
        data_loader = next(
            (s for s in self.train_steps if s.type == "data_loader"), None
        )
        if data_loader:
            steps.append(data_loader)

        # Extract components
        preprocessors = self.extractor.extract_preprocessors(self.train_steps)
        producers = self.extractor.extract_model_producers(self.train_steps)

        # MLflow loader
        self.loader_builder.add_mlflow_loader(
            steps,
            init_id,
            alias,
            preprocessors,
            producers,
            preprocessor_step,
            feature_pipeline,
        )

        # Legacy preprocessor
        self._add_legacy_preprocessor(
            steps, preprocessor_step, init_id, preprocessors, data_loader
        )

        # Regular preprocessors
        self._add_preprocessors(steps, alias, init_id, data_loader)

        # Feature inference steps (NEW)
        if feature_pipeline:
            self._add_feature_inference_steps(steps, feature_pipeline, init_id)

        # Sub-pipelines and branches
        special, special_ids, branch_ids = self._process_special_steps(
            alias, feature_pipeline
        )
        steps.extend(special)

        # Evaluators
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

        # Auxiliary steps
        self._add_auxiliary_steps(steps, evaluator_ids)

        return steps

    def _add_feature_inference_steps(
        self,
        steps: List[Any],
        feature_pipeline: FeaturePipeline,
        init_id: str,
    ) -> None:
        """Add feature inference steps for engineered features.

        Parameters
        ----------
        steps : List[Any]
            Pipeline steps to append to.
        feature_pipeline : FeaturePipeline
            Feature pipeline config.
        init_id : str
            Init artifacts step ID.
        """
        for feat in feature_pipeline.engineered:
            # Skip if inside sub-pipeline (handled separately)
            if feat.parent_pipeline:
                continue

            step = self._create_feature_inference_step(feat, init_id)
            steps.append(step)

            print(
                f"[EvalBuilder] Added feature inference: "
                f"{step['id']} -> {feat.output_key}"
            )

    def _create_feature_inference_step(
        self, feat: EngineeredFeature, init_id: str
    ) -> DictConfig:
        """Create feature inference step config.

        Parameters
        ----------
        feat : EngineeredFeature
            Feature to generate.
        init_id : str
            Init artifacts step ID.

        Returns
        -------
        DictConfig
            Feature inference step.
        """
        step_id = f"{feat.source_step_id}_inference"
        model_key = f"fitted_{feat.source_step_id}"

        return OmegaConf.create(
            {
                "id": step_id,
                "type": "feature_inference",
                "enabled": True,
                "depends_on": [init_id],
                "source_model_key": model_key,
                "base_features_key": "preprocessed_data",
                "output_key": feat.output_key,
                "apply_windowing": self._should_apply_windowing(feat.source_step_id),
            }
        )

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

    def _transform_sub_pipeline(
        self, step: Any, alias: str, feature_pipeline: Optional[FeaturePipeline]
    ) -> Any:
        """Transform sub-pipeline for eval mode.

        Parameters
        ----------
        step : Any
            Sub-pipeline step.
        alias : str
            Model alias.
        feature_pipeline : Optional[FeaturePipeline]
            Feature pipeline config.

        Returns
        -------
        Any
            Transformed step.
        """
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

        # Transform nested steps
        for sub_step in transformed.pipeline.steps:
            if sub_step.type == "preprocessor":
                self.transformer.transform_preprocessor(sub_step, alias)

            elif sub_step.type == "dynamic_adapter":
                self._transform_dynamic_adapter_in_sub(
                    sub_step, alias, feature_pipeline
                )

            elif sub_step.type in ["clustering", "model"]:
                self._transform_model_in_sub(sub_step, alias, feature_pipeline)

        return transformed

    def _transform_dynamic_adapter_in_sub(
        self,
        step: Any,
        alias: str,
        feature_pipeline: Optional[FeaturePipeline],
    ) -> None:
        """Transform dynamic adapter in sub-pipeline."""
        if not step.get("log_artifact", False):
            return
        _ = alias
        # Check if it's a feature producer
        is_feature = False
        if feature_pipeline:
            is_feature = any(
                f.source_step_id == step["id"] for f in feature_pipeline.engineered
            )

        if is_feature:
            # Convert to feature inference mode
            step.is_train = False
            step.instance_key = f"fitted_{step['id']}"
            self.transformer.remove_training_configs(step)

            # Change run_method to transform
            if step.get("run_method") == "fit_transform":
                step.run_method = "transform"

    def _transform_model_in_sub(
        self,
        step: Any,
        alias: str,
        feature_pipeline: Optional[FeaturePipeline],
    ) -> None:
        """Transform model/clustering in sub-pipeline."""
        # Check if it's a feature producer
        is_feature = False
        _ = alias
        if feature_pipeline:
            is_feature = any(
                f.source_step_id == step["id"] for f in feature_pipeline.engineered
            )

        if is_feature:
            # Convert to inference mode
            step.type = "feature_inference"
            step.source_model_key = f"fitted_{step['id']}"
            step.base_features_key = "preprocessed_data"

            # Get output key from feature pipeline
            if feature_pipeline:
                for feat in feature_pipeline.engineered:
                    if feat.source_step_id == step["id"]:
                        step.output_key = feat.output_key
                        break

    def _process_special_steps(
        self, alias: str, feature_pipeline: Optional[FeaturePipeline]
    ) -> Tuple[List[Any], List[str], Set[str]]:
        """Process sub-pipelines and branches."""
        special = []
        special_ids = []
        branch_ids: Set[str] = set()

        for step in self.train_steps:
            if step.type == "sub_pipeline":
                transformed = self._transform_sub_pipeline(
                    step, alias, feature_pipeline
                )
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

    # Keep existing methods from original EvalBuilder
    def _add_legacy_preprocessor(
        self,
        steps: List[Any],
        preprocessor_step: Optional[Any],
        init_id: str,
        all_preprocessors: List[Any],
        data_loader: Optional[Any],
    ) -> None:
        """Add legacy preprocessor if needed."""
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
        """Add preprocessors to pipeline."""
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
        """Check if adapter should be added."""
        if not hasattr(step, "log_artifact") or not step.log_artifact:
            return False

        if not hasattr(step, "artifact_type"):
            return False

        return step.artifact_type == "preprocess"

    def _transform_branch(self, step: Any) -> Any:
        """Transform branch for eval."""
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
        """Convert branch to evaluator."""
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
        """Create evaluator wiring."""
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
        """Add evaluators."""
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
        """Check if clustering adapter."""
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
        """Create evaluator config."""
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
        """Build evaluator inputs."""
        inputs: Dict[str, Any] = {
            "features": features,
            "model": f"fitted_{mp_id}",
        }

        if not is_clustering:
            inputs["targets"] = "target_data"

        return inputs

    def _add_auxiliary_steps(self, steps: List[Any], evaluator_ids: List[str]) -> None:
        """Add logger and profiling."""
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
