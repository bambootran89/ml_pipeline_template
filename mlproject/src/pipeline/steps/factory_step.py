"""Factory for creating pipeline steps from configuration.

Enhanced với support cho advanced step types:
- parallel: Execute branches concurrently
- branch: Conditional execution
- sub_pipeline: Nested pipeline execution
- generic_model: Unified model step với wiring
- clustering: Clustering với auto output_as_feature

Pipeline executor with dependency resolution.

Enhanced to support runtime context pre-initialization for serving mode.

Advanced pipeline steps for complex workflows.

This module provides steps for:
- Parallel execution of multiple branches
- Conditional branching based on context values
- Sub-pipeline execution as single step
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Type

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.data_loader_step import DataLoaderStep
from mlproject.src.pipeline.steps.evaluator_step import EvaluatorStep
from mlproject.src.pipeline.steps.generic_model_step import (
    ClusteringModelStep,
    GenericModelStep,
)
from mlproject.src.pipeline.steps.inference_step import InferenceStep
from mlproject.src.pipeline.steps.logger_step import LoggerStep
from mlproject.src.pipeline.steps.model_loader_step import ModelLoaderStep
from mlproject.src.pipeline.steps.preprocessor_step import PreprocessorStep
from mlproject.src.pipeline.steps.trainer_step import TrainerStep
from mlproject.src.pipeline.steps.tuner_step import TunerStep


class PipelineExecutor:
    """Execute pipeline steps in dependency order.

    This executor:
    - Resolves step dependencies using topological sort
    - Manages shared context across steps
    - Handles step enablement and skipping
    - Supports runtime context pre-initialization (serving mode)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize pipeline executor.

        Parameters
        ----------
        cfg : DictConfig
            Full configuration including pipeline definition.
        """
        self.cfg = cfg
        self.steps: List[BasePipelineStep] = []
        self._build_steps()

    def _build_steps(self) -> None:
        """Build step instances from configuration."""
        pipeline_cfg = self.cfg.get("pipeline", {})
        step_configs = pipeline_cfg.get("steps", [])

        for step_config in step_configs:
            step = StepFactory.create(step_config, self.cfg)
            self.steps.append(step)

        print(f"[Pipeline] Built {len(self.steps)} steps")

    def _topological_sort(self) -> List[BasePipelineStep]:
        """Sort steps by dependencies using Kahn's algorithm.

        Returns
        -------
        List[BasePipelineStep]
            Steps in execution order.

        Raises
        ------
        RuntimeError
            If circular dependencies detected.
        """
        # Build adjacency list and in-degree count
        step_map = {s.step_id: s for s in self.steps}
        in_degree = {s.step_id: 0 for s in self.steps}
        adjacency: Dict[str, List[str]] = {s.step_id: [] for s in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_map:
                    raise RuntimeError(
                        f"Step '{step.step_id}' depends on unknown " f"step '{dep}'"
                    )
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        # Kahn's algorithm
        queue = [s.step_id for s in self.steps if in_degree[s.step_id] == 0]
        sorted_ids: List[str] = []

        while queue:
            current_id = queue.pop(0)
            sorted_ids.append(current_id)

            for neighbor_id in adjacency[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(sorted_ids) != len(self.steps):
            raise RuntimeError("Circular dependency detected in pipeline")

        return [step_map[step_id] for step_id in sorted_ids]

    def execute(
        self, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute all enabled steps in dependency order.

        Enhanced to support runtime context pre-initialization for serving mode.

        Parameters
        ----------
        initial_context : Dict[str, Any], optional
            Pre-initialized context for serving mode.
            If provided, these keys are available before first step executes.

            Common use case: serving mode where data is loaded outside pipeline
            - df: Input dataframe
            - train_df: Empty (not used in serving)
            - test_df: Same as df (will be preprocessed)
            - is_splited_input: False

            Default is None (backward compatible with existing usage).

        Returns
        -------
        Dict[str, Any]
            Final pipeline context with all outputs.

        Examples
        --------
        # Training mode (normal - backward compatible)
        >>> executor = PipelineExecutor(cfg)
        >>> context = executor.execute()

        # Serving mode (with pre-initialized context)
        >>> executor = PipelineExecutor(cfg)
        >>> initial_ctx = {"df": df, "test_df": df, "is_splited_input": False}
        >>> context = executor.execute(initial_context=initial_ctx)

        Notes
        -----
        - When initial_context is None, starts with empty dict (backward compatible)
        - When initial_context is provided, those keys are available to all steps
        - Steps execute in topological order based on dependencies
        """
        sorted_steps = self._topological_sort()

        # Initialize context (use provided context or empty dict)
        # This is the ONLY change to existing code!
        context: Dict[str, Any] = initial_context.copy() if initial_context else {}

        print("[Pipeline] Execution order:")
        for i, step in enumerate(sorted_steps, 1):
            status = "enabled" if step.enabled else "disabled"
            print(f"  {i}. {step.step_id} ({status})")

        print("\n[Pipeline] Executing steps...")

        for step in sorted_steps:
            if not step.enabled:
                print(f"[Pipeline] Skipping disabled step: {step.step_id}")
                continue

            print(f"\n[Pipeline] Executing: {step.step_id}")
            context = step.execute(context)

        print("\n[Pipeline] Execution complete")
        return context


class ParallelStep(BasePipelineStep):
    """
    Execute multiple step branches in parallel.

    This step enables concurrent execution of independent workflows,
    useful for ensemble training or parallel feature engineering.

    Context Inputs
    --------------
    All inputs from parent context are passed to each branch.

    Context Outputs
    ---------------
    Outputs from all branches are merged into parent context.
    Merge strategy determines how conflicts are handled.

    Configuration
    -------------
    branches : List[Dict]
        List of step configurations to execute in parallel.
    max_workers : int, default=4
        Maximum concurrent threads.
    merge_strategy : str, default="merge"
        How to combine branch outputs:
        - "merge": Combine all outputs (later overwrites earlier)
        - "collect": Nest outputs under branch ID keys

    Examples
    --------
    YAML configuration::

        - id: "ensemble_train"
          type: "parallel"
          depends_on: ["preprocess"]
          branches:
            - id: "xgb_branch"
              type: "trainer"
            - id: "cat_branch"
              type: "trainer"
          max_workers: 3
          merge_strategy: "collect"
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        branches: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = 4,
        merge_strategy: str = "merge",
        **kwargs: Any,
    ) -> None:
        """
        Initialize parallel step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Parent configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        branches : Optional[List[Dict]], default=None
            List of step configurations for parallel execution.
        max_workers : int, default=4
            Maximum concurrent threads.
        merge_strategy : str, default="merge"
            Output merge strategy ("merge" or "collect").
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.branches = branches or []
        self.max_workers = max_workers
        self.merge_strategy = merge_strategy

    def _execute_branch(
        self,
        branch_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a single branch with isolated context.

        Parameters
        ----------
        branch_config : Dict[str, Any]
            Step configuration for this branch.
        context : Dict[str, Any]
            Copy of parent context.

        Returns
        -------
        Dict[str, Any]
            Branch execution result.
        """
        # Import here to avoid circular imports

        branch_ctx = deepcopy(context)
        step = StepFactory.create(branch_config, self.cfg)
        result = step.execute(branch_ctx)

        return {
            "branch_id": branch_config.get("id", "unknown"),
            "result": result,
        }

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all branches in parallel.

        Parameters
        ----------
        context : Dict[str, Any]
            Parent pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with merged branch outputs.
        """
        self.validate_dependencies(context)

        if not self.branches:
            print(f"[{self.step_id}] No branches configured, skipping")
            return context

        branch_count = len(self.branches)
        print(
            f"[{self.step_id}] Executing {branch_count} branches in parallel "
            f"(max_workers={self.max_workers})"
        )

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_branch, branch, context): branch
                for branch in self.branches
            }

            for future in as_completed(futures):
                branch = futures[future]
                branch_id = branch.get("id", "unknown")
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[{self.step_id}] Branch '{branch_id}' completed")
                except Exception as e:
                    print(f"[{self.step_id}] Branch '{branch_id}' failed: {e}")
                    raise

        # Merge results
        merged = self._merge_results(context, results)
        print(f"[{self.step_id}] All branches completed and merged")

        return merged

    def _merge_results(
        self,
        context: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge branch results into parent context.

        Parameters
        ----------
        context : Dict[str, Any]
            Original parent context.
        results : List[Dict[str, Any]]
            Results from all branches.

        Returns
        -------
        Dict[str, Any]
            Merged context.
        """
        merged = context.copy()

        if self.merge_strategy == "collect":
            # Nest outputs under branch IDs
            for result in results:
                branch_id = result["branch_id"]
                branch_ctx = result["result"]

                # Store new keys under branch namespace
                for key, value in branch_ctx.items():
                    if key not in context:
                        merged[f"{branch_id}.{key}"] = value

        else:  # "merge" strategy
            # Simple merge, later branches overwrite earlier
            for result in results:
                branch_ctx = result["result"]
                for key, value in branch_ctx.items():
                    if key not in context:
                        merged[key] = value

        return merged


class BranchStep(BasePipelineStep):
    """
    Conditional execution based on context values.

    This step evaluates a condition and executes either the
    if_true or if_false branch accordingly.

    Configuration
    -------------
    condition : Dict
        Condition specification:
        - key: Context key to check
        - operator: Comparison operator (>, <, ==, !=, >=, <=, in, not_in)
        - value: Value to compare against
    if_true : Dict
        Step configuration to execute if condition is True.
    if_false : Dict, optional
        Step configuration to execute if condition is False.

    Examples
    --------
    YAML configuration::

        - id: "model_selection"
          type: "branch"
          depends_on: ["preprocess"]
          condition:
            key: "data_size"
            operator: ">"
            value: 10000
          if_true:
            id: "train_deep"
            type: "trainer"
          if_false:
            id: "train_simple"
            type: "trainer"
    """

    OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
        ">": operator.gt,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        "<=": operator.le,
        "in": lambda a, b: a in b,
        "not_in": lambda a, b: a not in b,
    }

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        condition: Optional[Dict[str, Any]] = None,
        if_true: Optional[Dict[str, Any]] = None,
        if_false: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize branch step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Parent configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        condition : Optional[Dict], default=None
            Condition specification with key, operator, value.
        if_true : Optional[Dict], default=None
            Step config for true branch.
        if_false : Optional[Dict], default=None
            Step config for false branch.
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.condition = condition or {}
        self.if_true = if_true
        self.if_false = if_false

    def _evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        bool
            Condition result.
        """
        if not self.condition:
            return True

        key = self.condition.get("key", "")
        op_name = self.condition.get("operator", "==")
        expected = self.condition.get("value")

        # Direct context access for dynamic key
        actual = context.get(key)

        if op_name not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {op_name}")

        op_func = self.OPERATORS[op_name]

        try:
            result = op_func(actual, expected)
        except TypeError:
            # Handle None comparison
            result = False

        print(
            f"[{self.step_id}] Condition: {key} {op_name} {expected} "
            f"(actual={actual}) -> {result}"
        )

        return result

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute appropriate branch based on condition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context after branch execution.
        """
        self.validate_dependencies(context)

        # Import here to avoid circular imports

        condition_result = self._evaluate_condition(context)

        if condition_result and self.if_true:
            print(f"[{self.step_id}] Executing TRUE branch")
            step = StepFactory.create(self.if_true, self.cfg)
            return step.execute(context)
        elif not condition_result and self.if_false:
            print(f"[{self.step_id}] Executing FALSE branch")
            step = StepFactory.create(self.if_false, self.cfg)
            return step.execute(context)
        else:
            print(f"[{self.step_id}] No branch to execute")
            return context


class SubPipelineStep(BasePipelineStep):
    """
    Execute a nested pipeline as a single step.

    This step enables hierarchical pipeline composition, allowing
    complex workflows to be encapsulated as reusable components.

    Configuration
    -------------
    pipeline : Dict
        Inline pipeline configuration with steps list.
    config_path : str, optional
        Path to external YAML file defining the sub-pipeline.
    output_prefix : str, optional
        Prefix for all output keys from sub-pipeline.
    isolated : bool, default=False
        If True, sub-pipeline starts with empty context.

    Examples
    --------
    Inline sub-pipeline::

        - id: "feature_engineering"
          type: "sub_pipeline"
          depends_on: ["load_data"]
          wiring:
            inputs:
              df: "raw_data"
          pipeline:
            steps:
              - id: "normalize"
                type: "preprocessor"
              - id: "cluster"
                type: "trainer"
                output_as_feature: true
          output_prefix: "feat_"
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        output_prefix: str = "",
        isolated: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize sub-pipeline step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Parent experiment configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        pipeline : Optional[Dict], default=None
            Inline pipeline configuration.
        config_path : Optional[str], default=None
            Path to external pipeline YAML.
        output_prefix : str, default=""
            Prefix for sub-pipeline outputs.
        isolated : bool, default=False
            If True, start with minimal context.
        **kwargs
            Additional parameters including wiring.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.pipeline_config = pipeline
        self.config_path = config_path
        self.output_prefix = output_prefix
        self.isolated = isolated

    def _build_sub_executor(self) -> PipelineExecutor:
        """
        Build executor for nested pipeline.

        Returns
        -------
        PipelineExecutor
            Executor for sub-pipeline.
        """

        if self.config_path:
            sub_cfg = OmegaConf.load(self.config_path)
            merged = OmegaConf.merge(self.cfg, sub_cfg)

        elif self.pipeline_config:
            sub_cfg = OmegaConf.create({"pipeline": self.pipeline_config})
            merged = OmegaConf.merge(self.cfg, sub_cfg)

        else:
            raise ValueError(
                f"Step '{self.step_id}' requires either \
                'pipeline_config' or 'config_path'."
            )

        # Ensure the merged config is a DictConfig (not ListConfig)
        if not isinstance(merged, DictConfig):
            container = OmegaConf.to_container(merged, resolve=True)
            if not isinstance(container, dict):
                raise TypeError(
                    f"Step '{self.step_id}': merged config must be dict/DictConfig, "
                    f"but received {type(container)}"
                )
            merged = OmegaConf.create(container)

        return PipelineExecutor(merged)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute nested pipeline.

        Parameters
        ----------
        context : Dict[str, Any]
            Parent pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with sub-pipeline outputs merged.
        """
        self.validate_dependencies(context)

        print(f"[{self.step_id}] Executing sub-pipeline")

        # Prepare input context
        if self.isolated:
            sub_ctx: Dict[str, Any] = {}
        else:
            sub_ctx = context.copy()

        # Apply input wiring (inherited from BasePipelineStep)
        for local_name, source_key in self.input_keys.items():
            if source_key in context:
                sub_ctx[local_name] = context[source_key]

        # Execute sub-pipeline
        executor = self._build_sub_executor()
        sub_result = executor.execute(initial_context=sub_ctx)

        # Merge outputs
        merged = context.copy()

        # Apply output wiring
        for local_name, target_key in self.output_keys.items():
            if local_name in sub_result:
                merged[target_key] = sub_result[local_name]

        # Copy new keys with prefix
        for key, value in sub_result.items():
            if key not in context and key not in self.output_keys:
                prefixed = f"{self.output_prefix}{key}"
                merged[prefixed] = value

        print(f"[{self.step_id}] Sub-pipeline completed")
        return merged


class StepFactory:
    """Factory for creating pipeline step instances.

    This factory maps step type strings to concrete step classes
    and handles instantiation with proper configuration.

    Supports both basic and advanced step types for complex
    pipeline workflows.

    Basic Types
    -----------
    - data_loader: Load data from source
    - preprocessor: Transform features
    - trainer: Train model (legacy)
    - evaluator: Evaluate model
    - inference: Generate predictions
    - model_loader: Load from MLflow
    - logger: Log to MLflow
    - tuner: Hyperparameter optimization

    Advanced Types
    --------------
    - parallel: Execute branches concurrently
    - branch: Conditional execution
    - sub_pipeline: Nested pipeline
    - generic_model: Unified model step với wiring
    - clustering: Clustering với auto output_as_feature
    """

    STEP_REGISTRY: Dict[str, Type[BasePipelineStep]] = {
        # Basic steps
        "data_loader": DataLoaderStep,
        "preprocessor": PreprocessorStep,
        "trainer": TrainerStep,
        "model_loader": ModelLoaderStep,
        "inference": InferenceStep,
        "evaluator": EvaluatorStep,
        "logger": LoggerStep,
        "tuner": TunerStep,
        # Advanced steps
        "parallel": ParallelStep,
        "branch": BranchStep,
        "sub_pipeline": SubPipelineStep,
        # Generic model steps
        "generic_model": GenericModelStep,
        "clustering": ClusteringModelStep,
    }

    @classmethod
    def create(cls, step_config: Dict[str, Any], cfg: DictConfig) -> BasePipelineStep:
        """Create a pipeline step from configuration.

        Parameters
        ----------
        step_config : Dict[str, Any]
            Step configuration with keys:
            - id: str (required)
            - type: str (required)
            - enabled: bool (optional, default=True)
            - depends_on: List[str] (optional)
            - wiring: Dict (optional) - input/output key mapping
            - **kwargs: Step-specific parameters
        cfg : DictConfig
            Full experiment configuration.

        Returns
        -------
        BasePipelineStep
            Instantiated step.

        Raises
        ------
        ValueError
            If step type is unknown.
        KeyError
            If required keys are missing.

        Examples
        --------
        Basic step::

            step_config = {
                "id": "train_model",
                "type": "trainer",
                "enabled": True,
                "depends_on": ["preprocess"]
            }

        Step with wiring::

            step_config = {
                "id": "train_xgb",
                "type": "generic_model",
                "wiring": {
                    "inputs": {"data": "kmeans_output"},
                    "outputs": {"model": "xgb_model"}
                }
            }

        Parallel step::

            step_config = {
                "id": "ensemble",
                "type": "parallel",
                "branches": [
                    {"id": "xgb", "type": "trainer"},
                    {"id": "cat", "type": "trainer"}
                ]
            }
        """
        step_id = step_config["id"]
        step_type = step_config["type"]
        enabled = step_config.get("enabled", True)
        depends_on = step_config.get("depends_on", [])

        if step_type not in cls.STEP_REGISTRY:
            raise ValueError(
                f"Unknown step type: {step_type}. "
                f"Available: {list(cls.STEP_REGISTRY.keys())}"
            )

        step_class = cls.STEP_REGISTRY[step_type]

        # Extract step-specific kwargs (including wiring)
        reserved_keys = {"id", "type", "enabled", "depends_on"}
        kwargs = {k: v for k, v in step_config.items() if k not in reserved_keys}

        return step_class(
            step_id=step_id,
            cfg=cfg,
            enabled=enabled,
            depends_on=depends_on,
            **kwargs,
        )

    @classmethod
    def register(cls, type_name: str, step_class: Type[BasePipelineStep]) -> None:
        """Register a new step type.

        Parameters
        ----------
        type_name : str
            Type string for YAML configuration.
        step_class : Type[BasePipelineStep]
            Step class to register.

        Examples
        --------
        ::

            class CustomStep(BasePipelineStep):
                def execute(self, context):
                    # Custom logic
                    return context

            StepFactory.register("custom", CustomStep)
        """
        cls.STEP_REGISTRY[type_name] = step_class

    @classmethod
    def available_types(cls) -> list:
        """Get list of available step types.

        Returns
        -------
        list
            Sorted list of registered step type names.
        """
        return sorted(cls.STEP_REGISTRY.keys())
