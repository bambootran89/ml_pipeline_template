from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

# pylint: disable=too-many-lines


class ConfigGenerator:
    """Expert pipeline configuration automation for eval/serve workloads.

    Loads a base training YAML configuration and generates transformed pipeline
    configs for evaluation or serving workloads. This generator does not modify
    training logic or saved MLflow artifacts, only re-wires dependencies.

    Supports:
    - Standard flat pipelines (standard_train.yaml)
    - Nested sub-pipelines (nested_suppipeline.yaml)
    - Two-stage pipelines (kmeans_then_xgboost.yaml)
    - Parallel ensemble pipelines
    """

    def __init__(self, train_config_path: str) -> None:
        """Initialize ConfigGenerator and load training configuration.

        Args:
            train_config_path: Path to the base training config YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        loaded = OmegaConf.load(train_config_path)
        assert isinstance(
            loaded, DictConfig
        ), f"Expected DictConfig but got {type(loaded).__name__}"
        self.train_cfg: DictConfig = loaded
        self.experiment_name: str = Path(train_config_path).stem

    # pylint: disable=too-many-branches
    def _extract_model_producers_recursive(  # noqa: C901
        self, steps: List[Any], _prefix: str = ""
    ) -> List[Any]:
        """Recursively extract model producers from steps including nested structures.

        Supports:
        - Direct model producers (trainer, clustering, framework_model)
        - Sub-pipelines with nested model producers
        - Branch steps with if_true/if_false model producers
        - Parallel steps with branches

        Args:
            steps: List of pipeline steps.
            prefix: Optional prefix for nested step IDs.

        Returns:
            List of model producer steps.
        """
        producers: List[Any] = []

        for step in steps:
            if step.type in ["trainer", "clustering", "framework_model"]:
                producers.append(step)
            elif step.type == "sub_pipeline":
                # Recursively extract from nested sub-pipeline
                if hasattr(step, "pipeline") and hasattr(step.pipeline, "steps"):
                    nested_producers = self._extract_model_producers_recursive(
                        step.pipeline.steps, _prefix=f"{step.id}_"
                    )
                    producers.extend(nested_producers)
            elif step.type == "branch":
                # Extract from if_true branch
                if hasattr(step, "if_true"):
                    if_true_step = step.if_true
                    if hasattr(if_true_step, "type") and if_true_step.type in [
                        "trainer",
                        "clustering",
                        "framework_model",
                    ]:
                        producers.append(if_true_step)

                # Extract from if_false branch
                if hasattr(step, "if_false"):
                    if_false_step = step.if_false
                    if hasattr(if_false_step, "type") and if_false_step.type in [
                        "trainer",
                        "clustering",
                        "framework_model",
                    ]:
                        producers.append(if_false_step)
            elif step.type == "parallel":
                # Extract from parallel branches
                if hasattr(step, "branches"):
                    for branch in step.branches:
                        if hasattr(branch, "type") and branch.type in [
                            "trainer",
                            "clustering",
                            "framework_model",
                        ]:
                            producers.append(branch)

        return producers

    def _extract_preprocessors_recursive(
        self, steps: List[Any], _prefix: str = ""
    ) -> List[Any]:
        """Recursively extract preprocessor steps from nested sub-pipelines.

        Args:
            steps: List of pipeline steps.
            prefix: Optional prefix for nested step IDs.

        Returns:
            List of preprocessor steps.
        """
        preprocessors: List[Any] = []

        for step in steps:
            if step.type == "preprocessor":
                preprocessors.append(step)
            elif step.type == "sub_pipeline":
                # Recursively extract from nested sub-pipeline
                if hasattr(step, "pipeline") and hasattr(step.pipeline, "steps"):
                    nested_preprocessors = self._extract_preprocessors_recursive(
                        step.pipeline.steps, _prefix=f"{step.id}_"
                    )
                    preprocessors.extend(nested_preprocessors)

        return preprocessors

    # pylint: disable=too-many-branches
    def _transform_sub_pipeline_for_eval(  # noqa: C901
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for eval mode.

        Args:
            sub_pipeline_step: The sub_pipeline step to transform.
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed sub_pipeline step for eval mode.
        """
        transformed = copy.deepcopy(sub_pipeline_step)

        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        # Transform steps inside sub-pipeline
        for step in transformed.pipeline.steps:
            if step.type == "preprocessor":
                # Set preprocessor to load mode
                step.is_train = False
                step.alias = alias
                step.instance_key = f"fitted_{step.id}"

                # Remove training-only configs
                if hasattr(step, "log_artifact"):
                    delattr(step, "log_artifact")
                if hasattr(step, "artifact_type"):
                    delattr(step, "artifact_type")
                if hasattr(step, "wiring"):
                    # Keep outputs but simplify inputs to use defaults
                    if "inputs" in step.wiring:
                        delattr(step.wiring, "inputs")

            elif step.type == "clustering":
                # Set clustering to inference mode
                if not hasattr(step, "wiring"):
                    step.wiring = OmegaConf.create({})
                if not hasattr(step.wiring, "inputs"):
                    step.wiring.inputs = OmegaConf.create({})
                if not hasattr(step.wiring, "outputs"):
                    step.wiring.outputs = OmegaConf.create({})

                # Wire to use loaded model
                step.wiring.inputs.model = f"fitted_{step.id}"
                step.wiring.inputs.features = "preprocessed_data"
                step.wiring.outputs.features = (
                    step.wiring.outputs.features
                    if hasattr(step.wiring.outputs, "features")
                    else "cluster_features"
                )
                step.wiring.outputs.model = f"fitted_{step.id}"

                # Remove training-only configs
                if hasattr(step, "log_artifact"):
                    delattr(step, "log_artifact")
                if hasattr(step, "artifact_type"):
                    delattr(step, "artifact_type")
                if hasattr(step, "hyperparams"):
                    delattr(step, "hyperparams")

        return transformed

    # pylint: disable=too-many-branches
    def _transform_sub_pipeline_for_serve(  # noqa: C901
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for serve mode.

        Args:
            sub_pipeline_step: The sub_pipeline step to transform.
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed sub_pipeline step for serve mode.
        """
        transformed = copy.deepcopy(sub_pipeline_step)

        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        # Remove depends_on load_data since data is pre-initialized in serve mode
        if hasattr(transformed, "depends_on"):
            transformed.depends_on = [
                dep for dep in transformed.depends_on if dep != "load_data"
            ]
            if not transformed.depends_on:
                delattr(transformed, "depends_on")

        # Transform steps inside sub-pipeline
        for step in transformed.pipeline.steps:
            if step.type == "preprocessor":
                # Set preprocessor to load mode
                step.is_train = False
                step.alias = alias
                step.instance_key = f"fitted_{step.id}"

                # Remove training-only configs
                if hasattr(step, "log_artifact"):
                    delattr(step, "log_artifact")
                if hasattr(step, "artifact_type"):
                    delattr(step, "artifact_type")
                if hasattr(step, "wiring"):
                    if "inputs" in step.wiring:
                        delattr(step.wiring, "inputs")

            elif step.type == "clustering":
                # For serve mode, clustering should do inference
                if not hasattr(step, "wiring"):
                    step.wiring = OmegaConf.create({})
                if not hasattr(step.wiring, "inputs"):
                    step.wiring.inputs = OmegaConf.create({})
                if not hasattr(step.wiring, "outputs"):
                    step.wiring.outputs = OmegaConf.create({})

                # Wire to use loaded model
                step.wiring.inputs.model = f"fitted_{step.id}"
                step.wiring.inputs.features = "preprocessed_data"
                step.wiring.outputs.features = (
                    step.wiring.outputs.features
                    if hasattr(step.wiring.outputs, "features")
                    else "cluster_features"
                )
                step.wiring.outputs.model = f"fitted_{step.id}"

                # Remove training-only configs
                if hasattr(step, "log_artifact"):
                    delattr(step, "log_artifact")
                if hasattr(step, "artifact_type"):
                    delattr(step, "artifact_type")
                if hasattr(step, "hyperparams"):
                    delattr(step, "hyperparams")

        return transformed

    def _transform_branch_step_for_eval(self, branch_step: Any, _alias: str) -> Any:
        """Transform a branch step for eval mode.

        Args:
            branch_step: The branch step to transform.
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed branch step for eval mode.
        """
        transformed = copy.deepcopy(branch_step)

        # Transform if_true branch: trainer/framework_model → evaluator
        if hasattr(transformed, "if_true"):
            if_true = transformed.if_true
            if if_true.type in ["trainer", "clustering", "framework_model"]:
                # Create evaluator ID based on model producer ID
                eval_id = (
                    f"{if_true.id}_evaluate"
                    if not if_true.id.endswith("_evaluate")
                    else if_true.id
                )

                # Build evaluator config
                eval_wiring = {
                    "inputs": {
                        "model": f"fitted_{if_true.id}",
                        "features": "preprocessed_data",
                    },
                    "outputs": {
                        "metrics": (
                            if_true.wiring.outputs.get("metrics", "evaluation_metrics")
                            if hasattr(if_true, "wiring")
                            and hasattr(if_true.wiring, "outputs")
                            else "evaluation_metrics"
                        )
                    },
                }

                # Add targets if not clustering
                if if_true.type != "clustering":
                    eval_wiring["inputs"]["targets"] = "target_data"

                # Create new evaluator step
                transformed.if_true = OmegaConf.create(
                    {
                        "id": eval_id,
                        "type": "evaluator",
                        "enabled": True,
                        "depends_on": (
                            ["preprocess"] if hasattr(transformed, "depends_on") else []
                        ),
                        "wiring": eval_wiring,
                    }
                )

                # Add step_eval_type for clustering
                if if_true.type == "clustering":
                    transformed.if_true.step_eval_type = "clustering"

        # Transform if_false branch: trainer/framework_model → evaluator
        if hasattr(transformed, "if_false"):
            if_false = transformed.if_false
            if if_false.type in ["trainer", "clustering", "framework_model"]:
                # Create evaluator ID based on model producer ID
                eval_id = (
                    f"{if_false.id}_evaluate"
                    if not if_false.id.endswith("_evaluate")
                    else if_false.id
                )

                # Build evaluator config
                eval_wiring = {
                    "inputs": {
                        "model": f"fitted_{if_false.id}",
                        "features": "preprocessed_data",
                    },
                    "outputs": {
                        "metrics": (
                            if_false.wiring.outputs.get("metrics", "evaluation_metrics")
                            if hasattr(if_false, "wiring")
                            and hasattr(if_false.wiring, "outputs")
                            else "evaluation_metrics"
                        )
                    },
                }

                # Add targets if not clustering
                if if_false.type != "clustering":
                    eval_wiring["inputs"]["targets"] = "target_data"

                # Create new evaluator step
                transformed.if_false = OmegaConf.create(
                    {
                        "id": eval_id,
                        "type": "evaluator",
                        "enabled": True,
                        "depends_on": (
                            ["preprocess"] if hasattr(transformed, "depends_on") else []
                        ),
                        "wiring": eval_wiring,
                    }
                )

                # Add step_eval_type for clustering
                if if_false.type == "clustering":
                    transformed.if_false.step_eval_type = "clustering"

        return transformed

    def _transform_branch_step_for_serve(self, branch_step: Any, _alias: str) -> Any:
        """Transform a branch step for serve mode.

        Args:
            branch_step: The branch step to transform.
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed branch step for serve mode.
        """
        transformed = copy.deepcopy(branch_step)

        # Transform if_true branch: trainer/framework_model → inference
        if hasattr(transformed, "if_true"):
            if_true = transformed.if_true
            if if_true.type in ["trainer", "clustering", "framework_model"]:
                # Create inference ID based on model producer ID
                inf_id = f"{if_true.id}_inference"

                # Build inference config
                inf_wiring = {
                    "inputs": {
                        "model": f"fitted_{if_true.id}",
                        "features": "preprocessed_data",
                    },
                    "outputs": {
                        "predictions": f"{if_true.id}_predictions",
                    },
                }

                # Create new inference step
                transformed.if_true = OmegaConf.create(
                    {
                        "id": inf_id,
                        "type": "inference",
                        "enabled": True,
                        "depends_on": ["preprocess"],
                        "output_as_feature": getattr(
                            if_true, "output_as_feature", False
                        ),
                        "wiring": inf_wiring,
                    }
                )

        # Transform if_false branch: trainer/framework_model → inference
        if hasattr(transformed, "if_false"):
            if_false = transformed.if_false
            if if_false.type in ["trainer", "clustering", "framework_model"]:
                # Create inference ID based on model producer ID
                inf_id = f"{if_false.id}_inference"

                # Build inference config
                inf_wiring = {
                    "inputs": {
                        "model": f"fitted_{if_false.id}",
                        "features": "preprocessed_data",
                    },
                    "outputs": {
                        "predictions": f"{if_false.id}_predictions",
                    },
                }

                # Create new inference step
                transformed.if_false = OmegaConf.create(
                    {
                        "id": inf_id,
                        "type": "inference",
                        "enabled": True,
                        "depends_on": ["preprocess"],
                        "output_as_feature": getattr(
                            if_false, "output_as_feature", False
                        ),
                        "wiring": inf_wiring,
                    }
                )

        return transformed

    def _make_evaluator_config(
        self,
        mp: Any,
        init_id: str,
        preprocessor_id: Optional[str],
    ) -> DictConfig:
        """Create evaluator config for a model producer step.

        Args:
            mp: Model producer step (trainer or clustering).
            init_id: MLflow loader step ID.
            preprocessor_id: Optional preprocessor step ID.

        Returns:
            DictConfig for evaluator step.
        """
        base_name = mp.id.replace("_features", "").replace("_model", "")
        eval_id = f"{base_name}_evaluate"

        inputs: Dict[str, Any] = {
            "features": "preprocessed_data",
            "model": f"fitted_{mp.id}",
        }
        if mp.type != "clustering":
            inputs["targets"] = "target_data"

        step_cfg: Dict[str, Any] = {
            "id": eval_id,
            "type": "evaluator",
            "enabled": True,
            "depends_on": [init_id],
            "wiring": {
                "inputs": inputs,
                "outputs": {"metrics": f"{base_name}_metrics"},
            },
        }

        if mp.type == "clustering":
            step_cfg["step_eval_type"] = "clustering"
        if preprocessor_id:
            step_cfg["depends_on"].append(preprocessor_id)

        return OmegaConf.create(step_cfg)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def _build_eval_steps(  # noqa: C901
        self,
        train_steps: List[Any],
        alias: str,
        init_id: str,
        _model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for eval pipeline, including evaluator and profiling.

        Supports nested sub-pipelines by keeping them intact but transforming
        their internal steps for evaluation mode.

        Args:
            train_steps: Original training pipeline steps.
            alias: MLflow alias for artifact loading.
            init_id: MLflow loader step ID.
            model_producers: Model producers from training steps (including nested).
            preprocessor_step: Optional top-level preprocessor step.

        Returns:
            New list of steps for eval mode.
        """
        new_steps: List[Any] = []
        evaluator_ids: List[str] = []

        # Find data loader
        data_loader = next((s for s in train_steps if s.type == "data_loader"), None)
        if data_loader:
            new_steps.append(data_loader)

        # Build load_map for all model producers and preprocessors (including nested)
        all_preprocessors = self._extract_preprocessors_recursive(train_steps)
        all_model_producers = self._extract_model_producers_recursive(train_steps)

        load_map = [
            {"step_id": mp.id, "context_key": f"fitted_{mp.id}"}
            for mp in all_model_producers
        ]

        # Add all preprocessors to load_map
        for prep in all_preprocessors:
            load_map.append(
                {
                    "step_id": prep.id,
                    "context_key": f"fitted_{prep.id}",
                }
            )

        # Special handling for top-level preprocessor (legacy compatibility)
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            load_map.append(
                {
                    "step_id": preprocessor_step.id,
                    "context_key": "transform_manager",
                }
            )

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

        # Handle top-level preprocessor (if exists and not in sub-pipeline)
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            prep = copy.deepcopy(preprocessor_step)
            prep.is_train = False
            prep.alias = alias
            prep.instance_key = "transform_manager"
            prep.depends_on = [init_id]
            prep.depends_on.append("load_data")
            new_steps.append(prep)

        # Process all steps, handling special types
        sub_pipeline_steps = []
        branch_steps = []
        parallel_steps = []
        branch_model_producer_ids = (
            set()
        )  # Track model producers inside branches/parallel

        # Handle top-level preprocessors (not in nested structures like sub_pipeline)
        for step in train_steps:
            if step.type == "preprocessor":
                # Check if this preprocessor is at top level (not in sub-pipeline)
                is_top_level = any(s.id == step.id for s in train_steps)
                if is_top_level:
                    prep = copy.deepcopy(step)
                    prep.is_train = False
                    prep.alias = alias
                    prep.instance_key = f"fitted_{step.id}"
                    prep.depends_on = [init_id]
                    if data_loader:
                        prep.depends_on.append("load_data")
                    new_steps.append(prep)

        # Handle sub-pipelines, branches, and parallel steps
        for step in train_steps:
            if step.type == "sub_pipeline":
                # Transform sub-pipeline for eval mode and keep it
                transformed_sub = self._transform_sub_pipeline_for_eval(step, alias)
                new_steps.append(transformed_sub)
                sub_pipeline_steps.append(step.id)
            elif step.type == "branch":
                # Transform branch for eval mode
                transformed_branch = self._transform_branch_step_for_eval(step, alias)
                new_steps.append(transformed_branch)
                branch_steps.append(step.id)

                # Track model producers in this branch (to avoid generating separate
                # evaluators)
                if hasattr(step, "if_true") and hasattr(step.if_true, "id"):
                    branch_model_producer_ids.add(step.if_true.id)
                if hasattr(step, "if_false") and hasattr(step.if_false, "id"):
                    branch_model_producer_ids.add(step.if_false.id)
            elif step.type == "parallel":
                # For parallel steps, DON'T track branches
                # They will get individual evaluators generated below
                parallel_steps.append(step.id)
            elif step.type in ["datamodule", "trainer", "framework_model"]:
                # Skip training steps in eval mode (they become evaluators or are in
                # branches/parallel)
                pass

        # Generate evaluators for model producers NOT in branches
        # Find preprocessor step ID for dependency
        preprocess_step_id = None
        for step in train_steps:
            if step.type == "preprocessor":
                preprocess_step_id = step.id
                break

        for mp in all_model_producers:
            if mp.id not in branch_model_producer_ids:
                ev = self._make_evaluator_config(
                    mp,
                    init_id,
                    preprocess_step_id,  # Add preprocess dependency
                )
                # If model producer is in sub-pipeline, evaluator depends on
                # sub-pipeline
                if sub_pipeline_steps:
                    ev.depends_on = [init_id] + sub_pipeline_steps
                elif preprocess_step_id:
                    # For parallel ensemble, add preprocess dependency
                    ev.depends_on = [init_id, preprocess_step_id]
                evaluator_ids.append(ev.id)
                new_steps.append(ev)

        # Collect evaluator IDs from branch steps (they're already in the branches)
        if branch_steps:
            evaluator_ids.extend(branch_steps)

        # Add profiling and logger steps
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

        return new_steps

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def _build_serve_steps(  # noqa: C901
        self,
        alias: str,
        init_id: str,
        model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for serve pipeline with inference chaining and profiling.

        Supports nested sub-pipelines by keeping them intact but transforming
        their internal steps for serving mode.

        Args:
            alias: MLflow alias for artifact loading.
            init_id: MLflow loader step ID.
            model_producers: Model producers from training steps (including nested).
            preprocessor_step: Optional preprocessor step.

        Returns:
            New list of steps for serve mode.
        """
        new_steps: List[Any] = []
        last_id: str = init_id

        # Get all preprocessors and model producers (including nested)
        train_steps = self.train_cfg.pipeline.steps
        all_preprocessors = self._extract_preprocessors_recursive(train_steps)
        all_model_producers = self._extract_model_producers_recursive(train_steps)

        # Build load_map for all artifacts
        load_map = [
            {"step_id": mp.id, "context_key": f"fitted_{mp.id}"}
            for mp in all_model_producers
        ]

        # Add all preprocessors to load_map
        for prep in all_preprocessors:
            load_map.append(
                {
                    "step_id": prep.id,
                    "context_key": f"fitted_{prep.id}",
                }
            )

        # Special handling for top-level preprocessor (legacy compatibility)
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            load_map.append(
                {
                    "step_id": preprocessor_step.id,
                    "context_key": "transform_manager",
                }
            )

        # MLflow loader step
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

        # Handle top-level preprocessors (not in nested structures like sub_pipeline)
        for step in train_steps:
            if step.type == "preprocessor":
                # Check if this preprocessor is at top level (not in sub-pipeline)
                is_top_level = any(s.id == step.id for s in train_steps)
                if is_top_level:
                    prep = copy.deepcopy(step)
                    prep.is_train = False
                    prep.alias = alias
                    prep.instance_key = f"fitted_{step.id}"
                    prep.depends_on = [init_id]
                    new_steps.append(prep)
                    last_id = prep.id

        # Handle top-level preprocessor (legacy compatibility)
        if preprocessor_step and not any(
            p.id == preprocessor_step.id for p in all_preprocessors
        ):
            prep = copy.deepcopy(preprocessor_step)
            prep.is_train = False
            prep.alias = alias
            prep.instance_key = "transform_manager"
            prep.depends_on = [init_id]
            new_steps.append(prep)
            last_id = prep.id

        # Check for special pipeline types
        has_sub_pipeline = any(s.type == "sub_pipeline" for s in train_steps)
        has_branch = any(s.type == "branch" for s in train_steps)
        has_parallel = any(s.type == "parallel" for s in train_steps)
        branch_model_producer_ids = set()

        # Handle sub-pipelines
        if has_sub_pipeline:
            for step in train_steps:
                if step.type == "sub_pipeline":
                    transformed_sub = self._transform_sub_pipeline_for_serve(
                        step, alias
                    )
                    new_steps.append(transformed_sub)
                    last_id = step.id

        # Handle branches
        if has_branch:
            for step in train_steps:
                if step.type == "branch":
                    transformed_branch = self._transform_branch_step_for_serve(
                        step, alias
                    )
                    new_steps.append(transformed_branch)
                    last_id = step.id

                    # Track model producers in this branch
                    if hasattr(step, "if_true") and hasattr(step.if_true, "id"):
                        branch_model_producer_ids.add(step.if_true.id)
                    if hasattr(step, "if_false") and hasattr(step.if_false, "id"):
                        branch_model_producer_ids.add(step.if_false.id)

        # Handle parallel steps - DON'T track branches
        # They will get individual inference steps generated below

        # Generate inference steps for model producers NOT in branches or
        # sub-pipelines
        if has_sub_pipeline or has_branch or has_parallel:
            # For parallel ensemble, generate inference for ALL parallel
            # branch producers. For other cases, only generate for
            # top-level model producers
            top_level_producers = []

            if has_parallel:
                # Include all model producers
                # (parallel branches are in all_model_producers)
                top_level_producers = [
                    mp
                    for mp in all_model_producers
                    if mp.id not in branch_model_producer_ids
                ]
            else:
                # Only top-level producers for sub-pipeline/branch cases
                top_level_producers = [
                    mp
                    for mp in all_model_producers
                    if mp.id not in branch_model_producer_ids
                    and any(s.id == mp.id for s in train_steps)
                ]

            for mp in top_level_producers:
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
        else:
            # Standard flat pipeline - create inference steps for all model producers
            for mp in model_producers:
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

        # Final profiling
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

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save transformed pipeline configuration to YAML file.

        Args:
            cfg: Transformed pipeline DictConfig.
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        print(f"[ConfigGenerator] Successfully generated: {path}")

    def _transform_pipeline(self, mode: str, alias: str) -> DictConfig:
        """Transform base training pipeline into eval or serve pipeline config.

        Args:
            mode: "eval" or "serve".
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed DictConfig for selected mode.

        Raises:
            ValueError: If pipeline.steps is missing.
        """
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(
                f"Config {self.experiment_name} is missing pipeline.steps definition"
            )

        train_steps = cfg.pipeline.steps
        model_producers = [
            s for s in train_steps if s.type in ["trainer", "clustering"]
        ]
        preprocessor_step = next(
            (s for s in train_steps if s.type == "preprocessor"), None
        )
        init_id = "init_artifacts"

        if mode == "eval":
            new_steps = self._build_eval_steps(
                train_steps, alias, init_id, model_producers, preprocessor_step
            )
        else:
            new_steps = self._build_serve_steps(
                alias, init_id, model_producers, preprocessor_step
            )

        cfg.pipeline.name = f"{self.experiment_name}_{mode}"
        cfg.pipeline.steps = new_steps

        if "preprocessing" in cfg:
            cfg.preprocessing.is_train = False

        return cfg

    def generate_all(self, output_dir: str, alias: str = "latest") -> Dict[str, str]:
        """Generate both eval and serve pipeline YAML configs and save to files.

        Args:
            output_dir: Directory to store generated configs.
            alias: MLflow alias for artifact loading.

        Returns:
            Dictionary containing saved config paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eval_path = str(out / f"{self.experiment_name}_eval.yaml")
        serve_path = str(out / f"{self.experiment_name}_serve.yaml")

        cfg_eval = self._transform_pipeline("eval", alias)
        cfg_serve = self._transform_pipeline("serve", alias)

        self._save_config(cfg_eval, eval_path)
        self._save_config(cfg_serve, serve_path)

        return {"eval": eval_path, "serve": serve_path}

    def generate_eval_config(self, alias: str, output_path: str) -> str:
        """Generate evaluation pipeline config and save to file.

        Args:
            alias: MLflow alias.
            output_path: Destination YAML path.

        Returns:
            Path to saved config.
        """
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config and save to file.

        Args:
            alias: MLflow alias.
            output_path: Destination YAML path.

        Returns:
            Path to saved config.
        """
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path
