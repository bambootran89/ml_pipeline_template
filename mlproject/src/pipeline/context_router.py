"""
Context Router for flexible data wiring between pipeline steps.

This module provides a unified mechanism for routing data between steps
using configurable input/output key mappings. It enables complex DAG
topologies without modifying individual step implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig


class ContextRouter:
    """
    Route data between pipeline steps using configurable key mappings.

    This class handles:
    - Resolving input data from context using flexible key patterns
    - Storing output data with configurable key names
    - Supporting nested key access (e.g., "step1.output.features")
    - Default fallback when keys are not specified

    Attributes
    ----------
    input_keys : Dict[str, str]
        Mapping of local names to context keys for inputs.
    output_keys : Dict[str, str]
        Mapping of local names to context keys for outputs.
    step_id : str
        Identifier of the owning step for default key generation.
    """

    def __init__(
        self,
        step_id: str,
        input_keys: Optional[Dict[str, str]] = None,
        output_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize context router.

        Parameters
        ----------
        step_id : str
            Unique step identifier for default key prefixing.
        input_keys : Optional[Dict[str, str]], default=None
            Input key mappings. Keys are local names, values are context keys.
            Example: {"data": "preprocessor_output", "model": "kmeans_model"}
        output_keys : Optional[Dict[str, str]], default=None
            Output key mappings. Keys are local names, values are context keys.
            Example: {"predictions": "xgb_predictions", "model": "xgb_model"}
        """
        self.step_id = step_id
        self.input_keys = input_keys or {}
        self.output_keys = output_keys or {}

    def get_input(
        self,
        context: Dict[str, Any],
        local_name: str,
        default_key: Optional[str] = None,
        required: bool = True,
    ) -> Any:
        """
        Retrieve input data from context using configured mapping.

        Resolution order:
        1. Check input_keys mapping for local_name
        2. Fall back to default_key if provided
        3. Fall back to "{step_id}_{local_name}" pattern

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context containing step outputs.
        local_name : str
            Local name used within the step (e.g., "data", "features").
        default_key : Optional[str], default=None
            Fallback context key if no mapping exists.
        required : bool, default=True
            Raise error if key not found.

        Returns
        -------
        Any
            Retrieved data from context.

        Raises
        ------
        KeyError
            If required=True and key not found in context.
        """
        # Determine actual context key
        if local_name in self.input_keys:
            context_key = self.input_keys[local_name]
        elif default_key:
            context_key = default_key
        else:
            context_key = f"{self.step_id}_{local_name}"

        # Handle nested key access (e.g., "step1.output.data")
        value = self._resolve_nested_key(context, context_key)

        if value is None and required:
            available = list(context.keys())
            raise KeyError(
                f"Input '{local_name}' (key='{context_key}') not found. "
                f"Available keys: {available}"
            )

        return value

    def set_output(
        self,
        context: Dict[str, Any],
        local_name: str,
        value: Any,
        default_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store output data in context using configured mapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context to update.
        local_name : str
            Local name used within the step.
        value : Any
            Data to store.
        default_key : Optional[str], default=None
            Fallback context key if no mapping exists.

        Returns
        -------
        Dict[str, Any]
            Updated context.
        """
        # Determine actual context key
        if local_name in self.output_keys:
            context_key = self.output_keys[local_name]
        elif default_key:
            context_key = default_key
        else:
            context_key = f"{self.step_id}_{local_name}"

        context[context_key] = value
        return context

    def _resolve_nested_key(
        self,
        context: Dict[str, Any],
        key: str,
    ) -> Any:
        """
        Resolve nested key access using dot notation.

        Parameters
        ----------
        context : Dict[str, Any]
            Context dictionary.
        key : str
            Key string, may contain dots for nested access.

        Returns
        -------
        Any
            Resolved value or None if not found.
        """
        if "." not in key:
            return context.get(key)

        parts = key.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

            if current is None:
                return None

        return current

    def get_all_inputs(
        self,
        context: Dict[str, Any],
        defaults: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve all configured inputs from context.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        defaults : Optional[Dict[str, str]], default=None
            Default key mappings for inputs not in input_keys.

        Returns
        -------
        Dict[str, Any]
            Dictionary of local_name -> value pairs.
        """
        defaults = defaults or {}
        result: Dict[str, Any] = {}

        all_keys = set(self.input_keys.keys()) | set(defaults.keys())

        for local_name in all_keys:
            default_key = defaults.get(local_name)
            try:
                value = self.get_input(context, local_name, default_key, required=False)
                if value is not None:
                    result[local_name] = value
            except KeyError:
                continue

        return result

    def set_all_outputs(
        self,
        context: Dict[str, Any],
        outputs: Dict[str, Any],
        defaults: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Store all outputs to context.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context to update.
        outputs : Dict[str, Any]
            Dictionary of local_name -> value pairs.
        defaults : Optional[Dict[str, str]], default=None
            Default key mappings for outputs not in output_keys.

        Returns
        -------
        Dict[str, Any]
            Updated context.
        """
        defaults = defaults or {}

        for local_name, value in outputs.items():
            default_key = defaults.get(local_name)
            self.set_output(context, local_name, value, default_key)

        return context


def parse_wiring_config(
    step_config: Union[Dict[str, Any], DictConfig],
) -> Dict[str, Any]:
    """
    Parse wiring configuration from step config.

    Extracts input_keys, output_keys from step configuration
    in a standardized format.

    Parameters
    ----------
    step_config : Union[Dict[str, Any], DictConfig]
        Step configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Parsed wiring config with keys:
        - input_keys: Dict[str, str]
        - output_keys: Dict[str, str]

    Examples
    --------
    YAML config::

        - id: "kmeans"
          type: "clustering"
          wiring:
            inputs:
              data: "preprocessor_output"
            outputs:
              labels: "cluster_labels"
              model: "kmeans_model"
    """
    wiring = step_config.get("wiring", {})

    input_keys: Dict[str, str] = {}
    output_keys: Dict[str, str] = {}

    # Parse inputs
    inputs_raw = wiring.get("inputs", {})
    if isinstance(inputs_raw, (dict, DictConfig)):
        for k, v in inputs_raw.items():
            input_keys[str(k)] = str(v)

    # Parse outputs
    outputs_raw = wiring.get("outputs", {})
    if isinstance(outputs_raw, (dict, DictConfig)):
        for k, v in outputs_raw.items():
            output_keys[str(k)] = str(v)

    # Also support shorthand: input_key, output_key (single key)
    if "input_key" in step_config:
        input_keys["data"] = str(step_config["input_key"])

    if "output_key" in step_config:
        output_keys["data"] = str(step_config["output_key"])

    return {
        "input_keys": input_keys,
        "output_keys": output_keys,
    }


def create_router_from_config(
    step_id: str,
    step_config: Union[Dict[str, Any], DictConfig],
) -> ContextRouter:
    """
    Create ContextRouter instance from step configuration.

    Parameters
    ----------
    step_id : str
        Unique step identifier.
    step_config : Union[Dict[str, Any], DictConfig]
        Step configuration with optional wiring section.

    Returns
    -------
    ContextRouter
        Configured router instance.
    """
    wiring = parse_wiring_config(step_config)

    return ContextRouter(
        step_id=step_id,
        input_keys=wiring["input_keys"],
        output_keys=wiring["output_keys"],
    )
