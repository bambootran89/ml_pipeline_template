"""Pipeline executor with dependency resolution.

Enhanced to support runtime context pre-initialization for serving mode.
"""

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from mlproject.src.pipeline.compat.v2.steps.base import BasePipelineStep
from mlproject.src.pipeline.compat.v2.steps.factory_step import StepFactory


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
