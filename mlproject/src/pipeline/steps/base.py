"""
Base interface for pipeline steps with integrated data wiring.

Defines the contract for all pipeline execution steps including
data loading, preprocessing, training, and evaluation.

Data Wiring cho phép flexible input/output key mapping giữa các steps
thông qua YAML configuration mà không cần thay đổi step code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig


class BasePipelineStep(ABC):
    """
    Abstract base class for pipeline steps với data wiring support.

    Each step must implement execute() and declare its dependencies.
    Steps form a DAG (Directed Acyclic Graph) for flexible execution.

    Data Wiring
    -----------
    Steps có thể configure custom input/output key mappings qua YAML:

        - id: "train_model"
          type: "trainer"
          wiring:
            inputs:
              data: "custom_preprocessed"  # Read from this key
            outputs:
              model: "my_model"            # Write to this key

    Attributes
    ----------
    step_id : str
        Unique identifier for this step.
    cfg : DictConfig
        Step-specific configuration.
    enabled : bool
        Whether this step is active.
    depends_on : List[str]
        List of step IDs that must complete before this step.
    input_keys : Dict[str, str]
        Mapping of local names to context keys for inputs.
    output_keys : Dict[str, str]
        Mapping of local names to context keys for outputs.
    """

    # Default context keys for backward compatibility
    # Subclasses can override these
    DEFAULT_INPUTS: Dict[str, str] = {}
    DEFAULT_OUTPUTS: Dict[str, str] = {}

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        wiring: Optional[Dict[str, Any]] = None,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize pipeline step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration for this step.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            IDs of prerequisite steps.
        wiring : Optional[Dict[str, Any]], default=None
            Input/output key mappings. Structure:
            {"inputs": {"local": "context_key"}, "outputs": {...}}
        input_key : Optional[str], default=None
            Shorthand for single input mapping (maps to "data").
        output_key : Optional[str], default=None
            Shorthand for single output mapping (maps to "data").
        **kwargs
            Step-specific parameters.
        """
        self.step_id = step_id
        self.cfg = cfg
        self.enabled = enabled
        self.depends_on = depends_on or []
        self._kwargs = kwargs

        # Parse wiring configuration
        self.input_keys, self.output_keys = self._parse_wiring(
            wiring, input_key, output_key
        )

    def _parse_wiring(
        self,
        wiring: Optional[Dict[str, Any]],
        input_key: Optional[str],
        output_key: Optional[str],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Parse wiring configuration into input/output key mappings.

        Parameters
        ----------
        wiring : Optional[Dict[str, Any]]
            Full wiring config with inputs/outputs.
        input_key : Optional[str]
            Shorthand single input key.
        output_key : Optional[str]
            Shorthand single output key.

        Returns
        -------
        Tuple[Dict[str, str], Dict[str, str]]
            (input_keys, output_keys) mappings.
        """
        input_keys: Dict[str, str] = {}
        output_keys: Dict[str, str] = {}

        # Process full wiring config
        if wiring:
            inputs_cfg = wiring.get("inputs", {})
            outputs_cfg = wiring.get("outputs", {})

            # Handle DictConfig or dict
            if hasattr(inputs_cfg, "items"):
                for k, v in inputs_cfg.items():
                    input_keys[str(k)] = str(v)
            if hasattr(outputs_cfg, "items"):
                for k, v in outputs_cfg.items():
                    output_keys[str(k)] = str(v)

        # Process shorthand keys
        if input_key:
            input_keys["data"] = str(input_key)
        if output_key:
            output_keys["data"] = str(output_key)

        return input_keys, output_keys

    def get_input(
        self,
        context: Dict[str, Any],
        local_name: str,
        default_key: Optional[str] = None,
        required: bool = True,
    ) -> Any:
        """
        Get input value from context using wiring configuration.

        Priority:
        1. Configured wiring (input_keys)
        2. Default key pattern (if provided)
        3. Class DEFAULT_INPUTS

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        local_name : str
            Local name for this input (e.g., "data", "model").
        default_key : Optional[str], default=None
            Fallback context key if not in wiring.
        required : bool, default=True
            If True, raises KeyError when missing.

        Returns
        -------
        Any
            Value from context, or None if not found and not required.

        Raises
        ------
        KeyError
            If required and key not found in context.
        """
        # Determine context key
        if local_name in self.input_keys:
            context_key = self.input_keys[local_name]
        elif default_key:
            context_key = default_key
        elif local_name in self.DEFAULT_INPUTS:
            context_key = self.DEFAULT_INPUTS[local_name]
        else:
            context_key = local_name

        # Handle nested keys (e.g., "step1.output.features")
        value = self._get_nested(context, context_key)

        if value is None and required:
            available = list(context.keys())
            raise KeyError(
                f"Step '{self.step_id}': Required input '{local_name}' "
                f"not found at key '{context_key}'. Available: {available}"
            )

        return value

    def set_output(
        self,
        context: Dict[str, Any],
        local_name: str,
        value: Any,
        default_key: Optional[str] = None,
    ) -> None:
        """
        Store output value in context using wiring configuration.

        Priority:
        1. Configured wiring (output_keys)
        2. Default key pattern (if provided)
        3. Pattern: {step_id}_{local_name}

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context to update.
        local_name : str
            Local name for this output.
        value : Any
            Value to store.
        default_key : Optional[str], default=None
            Fallback context key if not in wiring.
        """
        if local_name in self.output_keys:
            context_key = self.output_keys[local_name]
        elif default_key:
            context_key = default_key
        elif local_name in self.DEFAULT_OUTPUTS:
            context_key = self.DEFAULT_OUTPUTS[local_name]
        else:
            context_key = f"{self.step_id}_{local_name}"

        context[context_key] = value

    def _get_nested(self, data: Dict[str, Any], key: str) -> Any:
        """
        Get value from nested dict using dot notation.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary to search.
        key : str
            Key with optional dots for nesting (e.g., "a.b.c").

        Returns
        -------
        Any
            Found value or None.
        """
        if "." not in key:
            return data.get(key)

        parts = key.split(".")
        current: Union[Dict[str, Any], Any, None] = data  # annotate rộng hơn

        for part in parts:
            if not isinstance(current, dict):  # thu hẹp kiểu
                return None
            current = current.get(part)
            if current is None:
                return None
        return current

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this pipeline step.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline context containing outputs from
            previous steps.

        Returns
        -------
        Dict[str, Any]
            Updated context with this step's outputs.

        Raises
        ------
        RuntimeError
            If step execution fails.
        """
        raise NotImplementedError

    def validate_dependencies(self, context: Dict[str, Any]) -> None:
        """
        Validate that all dependencies are satisfied.

        Note: This only validates execution order, not context keys.
        Each step should validate its own required context keys in execute().

        Parameters
        ----------
        context : Dict[str, Any]
            Current pipeline context.
        """
        # Dependencies are satisfied by DAG execution order
