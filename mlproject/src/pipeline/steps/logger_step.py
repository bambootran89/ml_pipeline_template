"""
Discovery-based LoggerStep for systematic MLflow tracking.
"""

from __future__ import annotations

from typing import Any, Dict

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow


class LoggerStep(BasePipelineStep):
    """
    Systematic logger that discovers and persists all marked components.

    This step eliminates hardcoded keys by iterating through a central
    registry populated by previous pipeline steps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with MLflowManager for centralized tracking."""
        super().__init__(*args, **kwargs)
        self.mlflow_manager = MLflowManager(self.cfg)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Iterate through the registry and log every registered component."""
        exp_name = self.cfg.experiment.name
        run_name = f"{exp_name}_run"

        with self.mlflow_manager.start_run(run_name=run_name):
            # 1. Automated Discovery Logging
            registry = context.get("_artifact_registry", {})
            for step_id, artifact in registry.items():
                print(f"[LoggerStep] Auto-logging component from step: {step_id}")

                # Each artifact is wrapped via ArtifactPyFuncWrapper automatically
                self.mlflow_manager.log_component(
                    obj=artifact["obj"],
                    name=f"{exp_name}_{step_id}",
                    artifact_type=artifact["type"],
                )

            # 2. Log Metrics (Standardized discovery via context)
            metrics = context.get("evaluation_metrics", {})
            if metrics:
                safe_metrics = flatten_metrics_for_mlflow(metrics)
                self.mlflow_manager.log_metadata(metrics=safe_metrics)

        print(f"[{self.step_id}] Finished systematic logging to MLflow.")
        return context


# Register step type
StepFactory.register("logger", LoggerStep)
