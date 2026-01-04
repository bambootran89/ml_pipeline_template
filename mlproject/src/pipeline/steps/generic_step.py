"""
Generic pipeline step for dynamic execution with robust configuration handling.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Mapping, Optional, cast

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class GenericStep(BasePipelineStep):
    """
    Step that dynamically loads classes and maps inputs/outputs via wiring.
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        **extra: Any,
    ) -> None:
        super().__init__(step_id, cfg, enabled, depends_on)
        self.class_path: Any = extra.get("class_path")
        self.run_method: str = str(extra.get("run_method", "fit"))
        self.hyperparams: Any = extra.get("hyperparams", {})
        self.wiring: Any = extra.get("wiring", {})

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic logic with robust config and error handling."""
        self.validate_dependencies(context)

        if not isinstance(self.class_path, str):
            raise ValueError(f"Step '{self.step_id}': 'class_path' is missing.")

        # 1. Resolve hyperparams safely (handle DictConfig or plain dict)
        hparams: Dict[str, Any] = {}
        if self.hyperparams:
            if OmegaConf.is_config(self.hyperparams):
                container = OmegaConf.to_container(self.hyperparams, resolve=True)
                hparams = cast(
                    Dict[str, Any], container if isinstance(container, dict) else {}
                )
            elif isinstance(self.hyperparams, dict):
                hparams = cast(Dict[str, Any], self.hyperparams)

        # 2. Dynamic Import & Instantiation
        module_path, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_cls: Any = getattr(module, class_name)
        instance: Any = target_cls(**hparams)

        # 3. Wiring Inputs
        inputs_map: Mapping[str, str] = {}
        if self.wiring and isinstance(self.wiring.get("inputs"), Mapping):
            inputs_map = self.wiring.get("inputs")  # type: ignore

        method_args: Dict[str, Any] = {
            str(arg): context.get(str(ctx_key)) for arg, ctx_key in inputs_map.items()
        }

        # 4. Execution
        print(f"[GenericStep] {self.step_id} -> {class_name}.{self.run_method}")
        method: Any = getattr(instance, self.run_method)
        result: Any = method(**method_args)

        # 5. Wiring Outputs
        outputs_map: Mapping[str, str] = {}
        if self.wiring and isinstance(self.wiring.get("outputs"), Mapping):
            outputs_map = self.wiring.get("outputs")  # type: ignore

        self._map_outputs(context, result, outputs_map, instance)

        # Keep model instance for later steps
        context[f"{self.step_id}_instance"] = instance
        return context

    def _map_outputs(
        self,
        context: Dict[str, Any],
        result: Any,
        mapping: Mapping[str, str],
        instance: Any,
    ) -> None:
        """Map execution results or instance back to context."""
        if not mapping:
            context[self.step_id] = result
            return

        for out_name, ctx_key in mapping.items():
            value: Any = None

            if self.run_method == "fit" and out_name == "model":
                value = instance
            elif isinstance(result, dict):
                value = result.get(out_name)
            elif hasattr(result, out_name):
                value = getattr(result, out_name)
            else:
                value = result

            # Avoid assigning invalid float (NaN/inf) silently
            if isinstance(value, float) and not math.isfinite(value):
                continue

            context[str(ctx_key)] = value


# Register step
StepFactory.register("generic", GenericStep)
