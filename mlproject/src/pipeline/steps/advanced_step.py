"""Advanced pipeline steps for complex workflows.

This module provides steps for:
- Parallel execution of multiple branches
- Conditional branching based on context values
- Sub-pipeline execution as single step
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class ParallelStep(BasePipelineStep):
    """Execute multiple step branches in parallel."""

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        branches: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.branches = branches or []
        self.max_workers = max_workers

    def _execute_branch(
        self,
        branch_config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        branch_ctx = deepcopy(context)
        step = StepFactory.create(branch_config, self.cfg)
        result = step.execute(branch_ctx)
        return {"branch_id": branch_config.get("id", "unknown"), "result": result}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_dependencies(context)

        if not self.branches:
            print(f"[{self.step_id}] No branches configured, skipping")
            return context

        print(
            f"[{self.step_id}] Executing {len(self.branches)} branches "
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
        print("CUONG: 1-", context.get("_artifact_registry", {}))
        merged = self._merge_results(context, results)
        print("CUONG: 2-", merged.get("_artifact_registry", {}))
        print(f"[{self.step_id}] All branches completed")
        return merged

    def _merge_results(
        self, context: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged = context.copy()
        for result in results:
            for key, value in result["result"].items():
                if key not in context:
                    merged[key] = value
                else:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    else:
                        merged[key] = value

        return merged


class BranchStep(BasePipelineStep):
    """Conditional execution based on context values."""

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
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.condition = condition or {}
        self.if_true = if_true
        self.if_false = if_false

    def _evaluate_condition(self, context: Dict[str, Any]) -> bool:
        if not self.condition:
            return True

        key = self.condition.get("key", "")
        op_name = self.condition.get("operator", "==")
        expected = self.condition.get("value")
        actual = context.get(key)

        if op_name not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {op_name}")

        try:
            result = self.OPERATORS[op_name](actual, expected)
        except TypeError:
            result = False

        print(
            f"[{self.step_id}] Condition: {key} {op_name} {expected} "
            f"(actual={actual}) -> {result}"
        )
        return result

    def _check_model_availability(
        self, branch_config: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if required model exists in context for inference/evaluator steps.

        Returns:
            (available, model_key): True if model is available or not required,
                                   False if model is required but missing.
        """
        if not branch_config:
            return True, None

        step_type = branch_config.get("type", "")
        if step_type not in ["inference", "evaluator"]:
            return True, None

        # Check if step has wiring with model input
        wiring = branch_config.get("wiring", {})
        inputs = wiring.get("inputs", {})
        model_key = inputs.get("model")

        if model_key and model_key not in context:
            return False, model_key

        return True, model_key

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_dependencies(context)
        condition_result = self._evaluate_condition(context)

        # Determine primary and fallback branches based on condition
        if condition_result:
            primary_branch = self.if_true
            primary_name = "TRUE"
            fallback_branch = self.if_false
            fallback_name = "FALSE"
        else:
            primary_branch = self.if_false
            primary_name = "FALSE"
            fallback_branch = self.if_true
            fallback_name = "TRUE"

        # Try primary branch first
        if primary_branch:
            available, model_key = self._check_model_availability(primary_branch, context)
            if available:
                print(f"[{self.step_id}] Executing {primary_name} branch")
                step = StepFactory.create(primary_branch, self.cfg)
                return step.execute(context)
            else:
                print(
                    f"[{self.step_id}] {primary_name} branch requires model "
                    f"'{model_key}' which is not available in context"
                )

        # Try fallback branch if primary failed
        if fallback_branch:
            available, model_key = self._check_model_availability(fallback_branch, context)
            if available:
                print(
                    f"[{self.step_id}] Falling back to {fallback_name} branch "
                    f"(primary branch model not available)"
                )
                step = StepFactory.create(fallback_branch, self.cfg)
                return step.execute(context)
            else:
                print(
                    f"[{self.step_id}] {fallback_name} branch requires model "
                    f"'{model_key}' which is not available in context"
                )

        # Neither branch can execute
        available_keys = [k for k in context.keys() if "_model" in str(k)]
        raise RuntimeError(
            f"Step '{self.step_id}': Cannot execute conditional branch. "
            f"Neither branch has its required model available in context. "
            f"Available model keys: {available_keys}. "
            f"This typically happens when the condition evaluates differently "
            f"between training and serving. Consider training both models "
            f"separately before using conditional branching for serving."
        )


class SubPipelineStep(BasePipelineStep):
    """Execute a nested pipeline as a single step."""

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        pipeline: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        isolated: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.pipeline_config = pipeline
        self.config_path = config_path

        self.isolated = isolated

    def _build_sub_executor(self):
        """Build executor for nested pipeline."""
        from mlproject.src.pipeline.executor import (  # pylint: disable=C0415
            PipelineExecutor,
        )

        if self.config_path:
            sub_cfg = OmegaConf.load(self.config_path)
            merged = OmegaConf.merge(self.cfg, sub_cfg)
        elif self.pipeline_config:
            sub_cfg = OmegaConf.create({"pipeline": self.pipeline_config})
            merged = OmegaConf.merge(self.cfg, sub_cfg)
        else:
            raise ValueError(
                f"Step '{self.step_id}' requires 'pipeline' or 'config_path'."
            )

        if not isinstance(merged, DictConfig):
            container = OmegaConf.to_container(merged, resolve=True)
            merged = OmegaConf.create(container)

        return PipelineExecutor(merged)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_dependencies(context)
        print(f"[{self.step_id}] Executing sub-pipeline")

        sub_ctx = {} if self.isolated else context.copy()

        for local_name, source_key in self.input_keys.items():
            if source_key in context:
                sub_ctx[local_name] = context[source_key]

        executor = self._build_sub_executor()
        sub_result = executor.execute(initial_context=sub_ctx)

        merged = context.copy()
        for local_name, target_key in self.output_keys.items():
            if local_name in sub_result:
                merged[target_key] = sub_result[local_name]

        for key, value in sub_result.items():
            if key not in context and key not in self.output_keys:
                merged[key] = value
        print(f"[{self.step_id}] Sub-pipeline completed")
        return merged


# Register step types
StepFactory.register("parallel", ParallelStep)
StepFactory.register("branch", BranchStep)
StepFactory.register("sub_pipeline", SubPipelineStep)
