"""
Enhanced base interface for pipeline steps with data wiring support.

This module extends the original BasePipelineStep to support:
- Configurable input/output key mappings
- Automatic context routing via ContextRouter
- Backward compatibility with existing steps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from mlproject.src.pipeline.context_router import (
    ContextRouter,
    create_router_from_config,
)


class WiredPipelineStep(ABC):
    """
    Enhanced base class for pipeline steps with flexible data wiring.

    This class extends the original step interface to support:
    - Explicit input/output key configuration via YAML
    - Automatic routing through ContextRouter
    - Default key patterns for backward compatibility

    Attributes
    ----------
    step_id : str
        Unique identifier for this step.
    cfg : DictConfig
        Full experiment configuration.
    step_config : Dict[str, Any]
        Step-specific configuration from pipeline YAML.
    enabled : bool
        Whether this step is active.
    depends_on : List[str]
        List of step IDs that must complete before this step.
    router : ContextRouter
        Data routing helper for input/output management.

    Examples
    --------
    YAML configuration with explicit wiring::

        - id: "feature_engineer"
          type: "custom_step"
          enabled: true
          depends_on: ["preprocess"]
          wiring:
            inputs:
              data: "preprocessed_data"
              config: "feature_config"
            outputs:
              features: "engineered_features"
              metadata: "feature_metadata"
    """

    # Default input/output key patterns for backward compatibility
    DEFAULT_INPUTS: Dict[str, str] = {}
    DEFAULT_OUTPUTS: Dict[str, str] = {}

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        step_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize wired pipeline step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Full experiment configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            IDs of prerequisite steps.
        step_config : Optional[Dict[str, Any]], default=None
            Step-specific configuration for wiring.
        **kwargs
            Additional step-specific parameters.
        """
        self.step_id = step_id
        self.cfg = cfg
        self.enabled = enabled
        self.depends_on = depends_on or []
        self.step_config = step_config or {}
        self._kwargs = kwargs

        # Initialize router from step config
        self.router = create_router_from_config(step_id, self.step_config)

    def get_input(
        self,
        context: Dict[str, Any],
        local_name: str,
        required: bool = True,
    ) -> Any:
        """
        Retrieve input from context using router.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        local_name : str
            Local name for the input (e.g., "data", "model").
        required : bool, default=True
            Raise error if not found.

        Returns
        -------
        Any
            Retrieved input value.
        """
        default_key = self.DEFAULT_INPUTS.get(local_name)
        return self.router.get_input(context, local_name, default_key, required)

    def set_output(
        self,
        context: Dict[str, Any],
        local_name: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Store output in context using router.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        local_name : str
            Local name for the output.
        value : Any
            Value to store.

        Returns
        -------
        Dict[str, Any]
            Updated context.
        """
        default_key = self.DEFAULT_OUTPUTS.get(local_name)
        return self.router.set_output(context, local_name, value, default_key)

    def validate_dependencies(self, context: Dict[str, Any]) -> None:
        """
        Validate that required inputs are available.

        Parameters
        ----------
        context : Dict[str, Any]
            Current pipeline context.
        """
        # Check all configured input keys exist
        for local_name in self.router.input_keys:
            try:
                self.get_input(context, local_name, required=True)
            except KeyError as exc:
                raise RuntimeError(
                    f"Step '{self.step_id}' missing required input: {exc}"
                ) from exc

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this pipeline step.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline context.

        Returns
        -------
        Dict[str, Any]
            Updated context with this step's outputs.
        """
        raise NotImplementedError


class StepAdapter:
    """
    Adapter to add wiring support to existing BasePipelineStep instances.

    This allows gradual migration without rewriting existing steps.
    """

    def __init__(
        self,
        step: Any,
        input_keys: Optional[Dict[str, str]] = None,
        output_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Wrap existing step with routing capability.

        Parameters
        ----------
        step : Any
            Original BasePipelineStep instance.
        input_keys : Optional[Dict[str, str]], default=None
            Input key mappings.
        output_keys : Optional[Dict[str, str]], default=None
            Output key mappings.
        """
        self._step = step
        self.router = ContextRouter(
            step_id=step.step_id,
            input_keys=input_keys or {},
            output_keys=output_keys or {},
        )

        # Forward attributes
        self.step_id = step.step_id
        self.enabled = step.enabled
        self.depends_on = step.depends_on

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute wrapped step with input/output remapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Updated context.
        """
        # Remap inputs before execution
        remapped_context = self._remap_inputs(context)

        # Execute original step
        result = self._step.execute(remapped_context)

        # Remap outputs after execution
        return self._remap_outputs(context, result)

    def _remap_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remap input keys to expected names."""
        remapped = context.copy()

        for local_name, context_key in self.router.input_keys.items():
            if context_key in context:
                # Map to default expected key
                default_key = f"{self._step.step_id}_{local_name}"
                remapped[default_key] = context[context_key]

        return remapped

    def _remap_outputs(
        self,
        original_context: Dict[str, Any],
        result_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remap output keys to configured names."""
        final = original_context.copy()

        # Copy all results
        for key, value in result_context.items():
            if key not in original_context:
                final[key] = value

        # Apply output remapping
        for local_name, target_key in self.router.output_keys.items():
            source_key = f"{self._step.step_id}_{local_name}"
            if source_key in result_context:
                final[target_key] = result_context[source_key]

        return final
