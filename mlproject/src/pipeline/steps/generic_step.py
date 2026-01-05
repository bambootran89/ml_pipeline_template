"""
Generic pipeline step supporting instance injection, reuse, and dynamic execution.

This step provides:
- Reuse of preloaded or injected fitted instances from pipeline context.
- Dynamic class loading and instantiation via import path.
- Execution of configurable methods (e.g., fit, transform, predict).
- Output wiring back into pipeline context for downstream steps.
- Optional artifact registration for model discovery and logging.

Quality gates:
- Fully typed and compliant with mypy static analysis.
- Line length constrained to 88 characters maximum.
- Pylint-compatible formatting and structure.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class GenericStep(BasePipelineStep):
    """Executes logic on a class instance with support for reuse from context."""

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the step with metadata for reuse and artifact logging."""
        super().__init__(step_id, cfg, enabled, depends_on)
        self.class_path: Optional[str] = kwargs.get("class_path")
        self.run_method: str = kwargs.get("run_method", "fit")
        self.hyperparams: Dict[str, Any] = kwargs.get("hyperparams", {})
        self.wiring: Dict[str, Any] = kwargs.get("wiring", {})
        self.log_artifact: bool = kwargs.get("log_artifact", False)
        self.artifact_type: str = kwargs.get("artifact_type", "component")
        self.instance_key: Optional[str] = kwargs.get("instance_key")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the configured method, reusing an existing instance if available."""
        self.validate_dependencies(context)

        instance: Optional[Any] = None

        if self.instance_key and self.instance_key in context:
            instance = context[self.instance_key]
            print(
                f"[{self.step_id}] Reusing injected instance under key: "
                f"{self.instance_key}"
            )

        if instance is None:
            if not self.class_path:
                raise ValueError(
                    f"Step '{self.step_id}': No instance found and no class path "
                    "provided for instantiation."
                )
            instance = self._instantiate()

        inputs_map: Dict[str, str] = self.wiring.get("inputs", {})
        method_kwargs: Dict[str, Any] = {
            key: context.get(context_key) for key, context_key in inputs_map.items()
        }

        print(
            f"[{self.step_id}] Executing method "
            f"{type(instance).__name__}.{self.run_method}"
        )

        method = getattr(instance, self.run_method)
        result: Any = method(**method_kwargs)

        outputs_map: Dict[str, str] = self.wiring.get("outputs", {})
        self._map_outputs(context, result, outputs_map, instance)

        if self.log_artifact:
            print(
                f"[{self.step_id}] Registering artifact of type: {self.artifact_type}"
            )
            self.register_for_discovery(context, instance)

        return context

    def _instantiate(self) -> Any:
        """
        Load and instantiate a class dynamically from an import path.

        Returns
        -------
        Any
            Instantiated object from the resolved class path.

        Raises
        ------
        AssertionError
            If class_path is not defined.
        TypeError
            If resolved hyperparameters are not a dictionary.
        """
        assert self.class_path is not None

        module_path, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        hparams: dict[str, Any] = {}
        if self.hyperparams:
            raw_params = OmegaConf.to_container(
                DictConfig(self.hyperparams),
                resolve=True,
            )

            if not isinstance(raw_params, dict):
                raise TypeError(
                    f"Step '{self.step_id}': Resolved hyperparameters must be a dict."
                )

            hparams = {str(k): v for k, v in raw_params.items()}

        return cls(**hparams)

    def _map_outputs(
        self,
        context: Dict[str, Any],
        result: Any,
        mapping: Dict[str, str],
        instance: Any,
    ) -> None:
        """Wire method outputs back into pipeline context."""
        if not mapping:
            context[self.step_id] = result
            return

        for out_name, ctx_key in mapping.items():
            if out_name == "model" and self.run_method == "fit":
                context[ctx_key] = instance
            elif isinstance(result, dict):
                context[ctx_key] = result.get(out_name)
            else:
                context[ctx_key] = result


StepFactory.register("generic", GenericStep)
