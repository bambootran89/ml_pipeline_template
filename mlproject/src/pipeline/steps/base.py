"""
Base interface for pipeline steps.

Defines the contract for all pipeline execution steps including
data loading, preprocessing, training, and evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig


class BasePipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Each step must implement execute() and declare its dependencies.
    Steps form a DAG (Directed Acyclic Graph) for flexible execution.

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
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        **kwargs,  # ← ADD: Accept step-specific parameters
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
        **kwargs
            Step-specific parameters (e.g., is_train, model_path, etc.)
        """
        self.step_id = step_id
        self.cfg = cfg
        self.enabled = enabled
        self.depends_on = depends_on or []
        # Store extra kwargs for subclasses to use
        self._kwargs = kwargs

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
        # No need to check context keys here as steps use different key patterns
