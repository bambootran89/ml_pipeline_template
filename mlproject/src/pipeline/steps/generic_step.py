"""
Generic pipeline step with automated artifact registration.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class GenericStep(BasePipelineStep):
    """
    Step that executes dynamic logic and registers instances for logging.

    This step allows identifying specific components that need to be
    persisted to MLflow by registering them in a central context registry.
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize step with artifact logging and wiring metadata."""
        super().__init__(step_id, cfg, enabled, depends_on)
        self.class_path: Optional[str] = kwargs.get("class_path")
        self.run_method: str = kwargs.get("run_method", "fit")
        self.hyperparams: Dict[str, Any] = kwargs.get("hyperparams", {})
        self.wiring: Dict[str, Any] = kwargs.get("wiring", {})
        # Expert feature: metadata for automated logging discovery
        self.log_artifact: bool = kwargs.get("log_artifact", False)
        self.artifact_type: str = kwargs.get("artifact_type", "component")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute logic and register the instance to context registry."""
        self.validate_dependencies(context)

        if not self.class_path:
            raise ValueError(f"Step '{self.step_id}': 'class_path' is missing.")

        instance = self._instantiate()
        inputs_map = self.wiring.get("inputs", {})
        method_kwargs = {k: context.get(v) for k, v in inputs_map.items()}

        print(f"[GenericStep] {self.step_id} -> Calling {self.run_method}")
        method = getattr(instance, self.run_method)
        result = method(**method_kwargs)

        # Map results to context
        outputs_map = self.wiring.get("outputs", {})
        self._map_outputs(context, result, outputs_map, instance)

        # Discovery Mechanism: Register for automated logging
        if self.log_artifact:
            self._register_for_discovery(context, instance)

        return context

    def _instantiate(self) -> Any:
        """Load class and instantiate with resolved hyperparams."""
        module_path, class_name = self.class_path.rsplit(".", 1)  # type: ignore
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        hparams = (
            OmegaConf.to_container(DictConfig(self.hyperparams), resolve=True)
            if self.hyperparams
            else {}
        )
        return cls(**hparams)

    def _register_for_discovery(self, context: Dict[str, Any], obj: Any) -> None:
        """Register the component into a central registry in the context."""
        if "_artifact_registry" not in context:
            context["_artifact_registry"] = {}

        # Save the instance and its intended artifact type
        context["_artifact_registry"][self.step_id] = {
            "obj": obj,
            "type": self.artifact_type,
        }

    def _map_outputs(
        self, context: Dict[str, Any], result: Any, mapping: Dict, ins: Any
    ) -> None:
        """Map execution results or the fitted instance back to context."""
        if not mapping:
            context[self.step_id] = result
            return
        for out_name, ctx_key in mapping.items():
            if out_name == "model" and self.run_method == "fit":
                context[ctx_key] = ins
            elif isinstance(result, dict):
                context[ctx_key] = result.get(out_name)
            else:
                context[ctx_key] = result


# Register step type
StepFactory.register("generic", GenericStep)
