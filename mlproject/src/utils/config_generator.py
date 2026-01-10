from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig, OmegaConf


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

    MODEL_PRODUCER_TYPES = ["trainer", "clustering", "framework_model"]

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
        """Recursively extract model producers from steps.

        Args:
            steps: List of pipeline steps.

        Returns:
            List of model producer steps.
        """
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
        """Recursively extract preprocessor steps.

        Args:
            steps: List of pipeline steps.

        Returns:
            List of preprocessor steps.
        """
        preprocessors: List[Any] = []
        for step in steps:
            if step.type == "preprocessor":
                preprocessors.append(step)
            elif step.type == "dynamic_adapter":
                # Support dynamic_adapter with artifact_type == "preprocess"
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

    def _transform_clustering_in_pipeline(self, step: Any, alias: str) -> None:
        """Transform clustering step in sub-pipeline."""
        self._setup_clustering_wiring(step)
        self._remove_training_configs(step)

    def _transform_sub_pipeline_for_eval(
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for eval mode."""
        transformed = copy.deepcopy(sub_pipeline_step)
        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        for step in transformed.pipeline.steps:
            if step.type == "preprocessor":
                self._transform_preprocessor_in_pipeline(step, alias)
            elif step.type == "clustering":
                self._transform_clustering_in_pipeline(step, alias)

        return transformed

    def _transform_sub_pipeline_for_serve(
        self, sub_pipeline_step: Any, alias: str
    ) -> Any:
        """Transform a sub_pipeline step for serve mode."""
        transformed = copy.deepcopy(sub_pipeline_step)
        if not hasattr(transformed, "pipeline") or not hasattr(
            transformed.pipeline, "steps"
        ):
            return transformed

        # Remove load_data dependency in serve mode
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

    def _make_evaluator_config(
        self,
        mp: Any,
        init_id: str,
        preprocessor_id: Optional[str],
    ) -> DictConfig:
        """Create evaluator config for a model producer step."""
        base_name = mp.id.replace("_features", "").replace("_model", "")
        eval_id = f"{base_name}_evaluate"

        # Check if this is a clustering model
        is_clustering = mp.type == "clustering"
        if mp.type == "dynamic_adapter" and hasattr(mp, "class_path"):
            # Check if class_path contains clustering-related keywords
            class_path_lower = mp.class_path.lower()
            is_clustering = (
                "cluster" in class_path_lower or "kmeans" in class_path_lower
            )

        inputs: Dict[str, Any] = {
            "features": "preprocessed_data",
            "model": f"fitted_{mp.id}",
        }
        if not is_clustering:
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

        if is_clustering:
            step_cfg["step_eval_type"] = "clustering"
        if preprocessor_id:
            step_cfg["depends_on"].append(preprocessor_id)

        return OmegaConf.create(step_cfg)

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
        """Process sub-pipelines, branches, and parallel steps.

        Returns:
            Tuple of (transformed_steps, step_ids, branch_producer_ids)
        """
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
                # Track branch producers
                for branch_name in ["if_true", "if_false"]:
                    if hasattr(step, branch_name):
                        branch = getattr(step, branch_name)
                        if hasattr(branch, "id"):
                            branch_producer_ids.add(branch.id)
            elif step.type == "parallel":
                special_ids.append(step.id)

        return special_steps, special_ids, branch_producer_ids

    def _build_eval_steps(
        self,
        train_steps: List[Any],
        alias: str,
        init_id: str,
        _unused_model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for eval pipeline."""
        new_steps: List[Any] = []

        # Add data loader
        data_loader = next((s for s in train_steps if s.type == "data_loader"), None)
        if data_loader:
            new_steps.append(data_loader)

        # Extract all producers and preprocessors
        all_preprocessors = self._extract_preprocessors_recursive(train_steps)
        all_model_producers = self._extract_model_producers_recursive(train_steps)

        # Build and add MLflow loader
        load_map = self._build_load_map(
            all_model_producers, all_preprocessors, preprocessor_step
        )
        self._add_mlflow_loader(new_steps, init_id, alias, load_map)

        # Add legacy preprocessor
        self._add_top_level_preprocessor_eval(
            new_steps, preprocessor_step, init_id, all_preprocessors, data_loader
        )

        # Add top-level preprocessors
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
                # Handle dynamic_adapter steps with artifact logging
                # Only add preprocessing adapters in eval mode
                # Model adapters are loaded via MLflow and used in evaluators
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
                    # Remove training-only configs
                    if hasattr(adapter, "log_artifact"):
                        delattr(adapter, "log_artifact")
                    if hasattr(adapter, "hyperparams"):
                        delattr(adapter, "hyperparams")
                    new_steps.append(adapter)

        # Process special steps
        special_steps, special_ids, branch_producer_ids = self._process_special_steps(
            train_steps, alias
        )
        new_steps.extend(special_steps)

        # Generate evaluators
        evaluator_ids = []
        preprocess_id = next(
            (s.id for s in train_steps if s.type == "preprocessor"), None
        )

        sub_pipeline_ids = [
            sid
            for sid, step in zip(special_ids, special_steps)
            if hasattr(step, "pipeline")
        ]

        for mp in all_model_producers:
            if mp.id not in branch_producer_ids:
                ev = self._make_evaluator_config(mp, init_id, preprocess_id)
                if sub_pipeline_ids:
                    ev.depends_on = [init_id] + sub_pipeline_ids
                elif preprocess_id:
                    ev.depends_on = [init_id, preprocess_id]
                evaluator_ids.append(ev.id)
                new_steps.append(ev)

        # Add branch evaluator IDs
        branch_ids = [
            sid
            for sid, step in zip(special_ids, special_steps)
            if hasattr(step, "condition")
        ]
        evaluator_ids.extend(branch_ids)

        # Add auxiliary steps
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

    def _build_serve_steps(
        self,
        alias: str,
        init_id: str,
        model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for serve pipeline."""
        new_steps: List[Any] = []
        train_steps = self.train_cfg.pipeline.steps

        # Extract all producers and preprocessors
        all_preprocessors = self._extract_preprocessors_recursive(train_steps)
        all_model_producers = self._extract_model_producers_recursive(train_steps)

        # Build and add MLflow loader
        load_map = self._build_load_map(
            all_model_producers, all_preprocessors, preprocessor_step
        )
        self._add_mlflow_loader(new_steps, init_id, alias, load_map)

        # Add preprocessors
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
                # Handle dynamic_adapter steps with artifact logging
                # Only add preprocessing adapters in serve mode
                # Model adapters are loaded via MLflow and used in inference
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
                    # Remove training-only configs
                    if hasattr(adapter, "log_artifact"):
                        delattr(adapter, "log_artifact")
                    if hasattr(adapter, "hyperparams"):
                        delattr(adapter, "hyperparams")
                    new_steps.append(adapter)
                    last_id = adapter.id

        # Legacy preprocessor
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

        # Check pipeline types
        has_sub_pipeline = any(s.type == "sub_pipeline" for s in train_steps)
        has_branch = any(s.type == "branch" for s in train_steps)
        has_parallel = any(s.type == "parallel" for s in train_steps)

        branch_producer_ids: Set[str] = set()

        # Handle sub-pipelines
        if has_sub_pipeline:
            for step in train_steps:
                if step.type == "sub_pipeline":
                    transformed = self._transform_sub_pipeline_for_serve(step, alias)
                    new_steps.append(transformed)
                    last_id = step.id

        # Handle branches
        if has_branch:
            for step in train_steps:
                if step.type == "branch":
                    transformed = self._transform_branch_step_for_serve(step)
                    new_steps.append(transformed)
                    last_id = step.id
                    for branch_name in ["if_true", "if_false"]:
                        if hasattr(step, branch_name):
                            branch = getattr(step, branch_name)
                            if hasattr(branch, "id"):
                                branch_producer_ids.add(branch.id)

        # Generate inference steps
        if has_sub_pipeline or has_branch or has_parallel:
            if has_parallel:
                producers = [
                    mp for mp in all_model_producers if mp.id not in branch_producer_ids
                ]
            else:
                producers = [
                    mp
                    for mp in all_model_producers
                    if mp.id not in branch_producer_ids
                    and any(s.id == mp.id for s in train_steps)
                ]
        else:
            producers = model_producers

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
        """Save transformed pipeline configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        print(f"[ConfigGenerator] Successfully generated: {path}")

    def _transform_pipeline(self, mode: str, alias: str) -> DictConfig:
        """Transform base training pipeline into eval or serve config."""
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(f"Config {self.experiment_name} missing pipeline.steps")

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
        """Generate both eval and serve pipeline YAML configs."""
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
        """Generate evaluation pipeline config and save to file."""
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config and save to file."""
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path
