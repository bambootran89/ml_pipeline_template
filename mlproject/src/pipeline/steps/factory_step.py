"""Factory for creating pipeline steps from configuration.

Enhanced with support for advanced step types:
- parallel: Execute branches concurrently
- branch: Conditional execution
- sub_pipeline: Nested pipeline execution
"""

from typing import Any, Dict, Type

from omegaconf import DictConfig

from mlproject.src.pipeline.steps.advanced_steps import (
    BranchStep,
    ParallelStep,
    SubPipelineStep,
)
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.data_loader_step import DataLoaderStep
from mlproject.src.pipeline.steps.evaluator_step import EvaluatorStep
from mlproject.src.pipeline.steps.inference_step import InferenceStep
from mlproject.src.pipeline.steps.logger_step import LoggerStep
from mlproject.src.pipeline.steps.model_loader_step import ModelLoaderStep
from mlproject.src.pipeline.steps.preprocessor_step import PreprocessorStep
from mlproject.src.pipeline.steps.trainer_step import TrainerStep
from mlproject.src.pipeline.steps.tuner_step import TunerStep


class StepFactory:
    """Factory for creating pipeline step instances.

    This factory maps step type strings to concrete step classes
    and handles instantiation with proper configuration.

    Supports both basic and advanced step types for complex
    pipeline workflows.

    Basic Types
    -----------
    - data_loader: Load data from source
    - preprocessor: Transform features
    - trainer: Train model
    - evaluator: Evaluate model
    - inference: Generate predictions
    - model_loader: Load from MLflow
    - logger: Log to MLflow
    - tuner: Hyperparameter optimization

    Advanced Types
    --------------
    - parallel: Execute branches concurrently
    - branch: Conditional execution
    - sub_pipeline: Nested pipeline
    """

    STEP_REGISTRY: Dict[str, Type[BasePipelineStep]] = {
        # Basic steps
        "data_loader": DataLoaderStep,
        "preprocessor": PreprocessorStep,
        "trainer": TrainerStep,
        "model_loader": ModelLoaderStep,
        "inference": InferenceStep,
        "evaluator": EvaluatorStep,
        "logger": LoggerStep,
        "tuner": TunerStep,
        # Advanced steps
        "parallel": ParallelStep,
        "branch": BranchStep,
        "sub_pipeline": SubPipelineStep,
    }

    @classmethod
    def create(cls, step_config: Dict[str, Any], cfg: DictConfig) -> BasePipelineStep:
        """Create a pipeline step from configuration.

        Parameters
        ----------
        step_config : Dict[str, Any]
            Step configuration with keys:
            - id: str (required)
            - type: str (required)
            - enabled: bool (optional, default=True)
            - depends_on: List[str] (optional)
            - wiring: Dict (optional) - input/output key mapping
            - **kwargs: Step-specific parameters
        cfg : DictConfig
            Full experiment configuration.

        Returns
        -------
        BasePipelineStep
            Instantiated step.

        Raises
        ------
        ValueError
            If step type is unknown.
        KeyError
            If required keys are missing.

        Examples
        --------
        Basic step::

            step_config = {
                "id": "train_model",
                "type": "trainer",
                "enabled": True,
                "depends_on": ["preprocess"]
            }

        Step with wiring::

            step_config = {
                "id": "train_xgb",
                "type": "trainer",
                "wiring": {
                    "inputs": {"data": "kmeans_output"},
                    "outputs": {"model": "xgb_model"}
                }
            }

        Parallel step::

            step_config = {
                "id": "ensemble",
                "type": "parallel",
                "branches": [
                    {"id": "xgb", "type": "trainer"},
                    {"id": "cat", "type": "trainer"}
                ]
            }
        """
        step_id = step_config["id"]
        step_type = step_config["type"]
        enabled = step_config.get("enabled", True)
        depends_on = step_config.get("depends_on", [])

        if step_type not in cls.STEP_REGISTRY:
            raise ValueError(
                f"Unknown step type: {step_type}. "
                f"Available: {list(cls.STEP_REGISTRY.keys())}"
            )

        step_class = cls.STEP_REGISTRY[step_type]

        # Extract step-specific kwargs (including wiring)
        reserved_keys = {"id", "type", "enabled", "depends_on"}
        kwargs = {k: v for k, v in step_config.items() if k not in reserved_keys}

        return step_class(
            step_id=step_id,
            cfg=cfg,
            enabled=enabled,
            depends_on=depends_on,
            **kwargs,
        )

    @classmethod
    def register(cls, type_name: str, step_class: Type[BasePipelineStep]) -> None:
        """Register a new step type.

        Parameters
        ----------
        type_name : str
            Type string for YAML configuration.
        step_class : Type[BasePipelineStep]
            Step class to register.

        Examples
        --------
        ::

            class CustomStep(BasePipelineStep):
                def execute(self, context):
                    # Custom logic
                    return context

            StepFactory.register("custom", CustomStep)
        """
        cls.STEP_REGISTRY[type_name] = step_class

    @classmethod
    def available_types(cls) -> list:
        """Get list of available step types.

        Returns
        -------
        list
            Sorted list of registered step type names.
        """
        return sorted(cls.STEP_REGISTRY.keys())
