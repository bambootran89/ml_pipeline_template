"""
Generic pipeline step with argument alignment and instance reuse.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class GenericStep(BasePipelineStep):
    """Dynamic pipeline step with intelligent parameter mapping and state reuse."""

    def __init__(self, step_id: str, cfg: DictConfig, **kwargs: Any) -> None:
        """Initialize dynamic step configuration and execution metadata."""
        super().__init__(step_id, cfg, **kwargs)

        self.class_path: Optional[str] = kwargs.get("class_path")
        self.run_method: str = kwargs.get("run_method", "fit")
        self.hyperparams: dict[str, Any] = kwargs.get("hyperparams", {})
        self.wiring: dict[str, Any] = kwargs.get("wiring", {})
        self.log_artifact: bool = kwargs.get("log_artifact", False)
        self.artifact_type: str = kwargs.get("artifact_type", "component")
        self.instance_key: Optional[str] = kwargs.get("instance_key")
        self.method_params: dict[str, Any] = kwargs.get("method_params", {})

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic step logic with automatic argument alignment."""
        self.validate_dependencies(context)

        # Resolve or create instance
        instance = context.get(self.instance_key) if self.instance_key else None
        if instance is None:
            instance = self._instantiate()
        else:
            print(f"[{self.step_id}] Reusing instance from context.")

        # Collect wired inputs
        inputs_map: dict[str, str] = self.wiring.get("inputs", {})
        raw_inputs: dict[str, Any] = {k: context.get(v) for k, v in inputs_map.items()}

        # Validate collected inputs
        for key, value in raw_inputs.items():
            if value is None:
                raise ValueError(
                    f"Step '{self.step_id}': Input '{key}' is None. Check wiring."
                )

        # Align inputs to method signature
        method = getattr(instance, self.run_method)
        sig = inspect.signature(method)
        aligned_kwargs = self._align_arguments(raw_inputs, dict(sig.parameters))

        # Add static method parameters if provided
        if self.method_params:
            static_opts = OmegaConf.to_container(
                DictConfig(self.method_params), resolve=True
            )
            if isinstance(static_opts, dict):
                aligned_kwargs.update({str(k): v for k, v in static_opts.items()})

        # Call method
        print(f"[{self.step_id}] Calling method '{self.run_method}'.")
        result = method(**aligned_kwargs)

        # Wire outputs back to context
        outputs_map: dict[str, str] = self.wiring.get("outputs", {})
        self._wire_outputs(context, result, instance, outputs_map)

        # Register for discovery if enabled
        if self.log_artifact:
            self.register_for_discovery(context, instance)

        return context

    @staticmethod
    def _align_arguments(
        inputs: dict[str, Any],
        parameters: dict[str, inspect.Parameter],
    ) -> dict[str, Any]:
        """Align common ML argument variants to match the target method signature."""
        aligned: dict[str, Any] = {}

        param_variants = {
            "X": ["X", "x", "data", "inputs"],
            "y": ["y", "Y", "target", "labels"],
            "sample_weight": ["sample_weight", "weights", "w"],
        }

        for param_name, param in parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                aligned.update(inputs)
                continue

            matched = False
            for variants in param_variants.values():
                if param_name in variants:
                    for v in variants:
                        if v in inputs:
                            aligned[param_name] = inputs[v]
                            matched = True
                            break
                    break

            if not matched and param_name in inputs:
                aligned[param_name] = inputs[param_name]

        return aligned

    def _instantiate(self) -> Any:
        """Dynamically import and instantiate a class using resolved hyperparameters."""
        assert self.class_path is not None

        module_path, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        init_params = OmegaConf.to_container(DictConfig(self.hyperparams), resolve=True)
        if not isinstance(init_params, dict):
            init_params = {}

        return cls(**{str(k): v for k, v in init_params.items()})

    def _wire_outputs(
        self,
        context: Dict[str, Any],
        result: Any,
        instance: Any,
        mapping: dict[str, str],
    ) -> None:
        """Write execution result or instance into context using wiring output map."""
        if not mapping:
            context[self.step_id] = result
            return

        for output_name, context_key in mapping.items():
            if output_name == "model" and self.run_method == "fit":
                context[context_key] = instance
            else:
                context[context_key] = result


StepFactory.register("generic", GenericStep)
