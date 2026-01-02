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
        """
        self.step_id = step_id
        self.cfg = cfg
        self.enabled = enabled
        self.depends_on = depends_on or []

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

        Parameters
        ----------
        context : Dict[str, Any]
            Current pipeline context.

        Raises
        ------
        ValueError
            If required dependencies are missing from context.
        """
        for dep in self.depends_on:
            if dep not in context:
                raise ValueError(
                    f"Step '{self.step_id}' requires "
                    f"'{dep}' but it's missing from context"
                )
