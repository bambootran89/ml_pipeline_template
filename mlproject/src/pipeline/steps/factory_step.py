"""Factory for creating pipeline steps from configuration."""

from typing import Any, Dict, Type

from omegaconf import DictConfig

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
    """

    STEP_REGISTRY: Dict[str, Type[BasePipelineStep]] = {
        "data_loader": DataLoaderStep,
        "preprocessor": PreprocessorStep,
        "trainer": TrainerStep,
        "model_loader": ModelLoaderStep,
        "inference": InferenceStep,
        "evaluator": EvaluatorStep,
        "logger": LoggerStep,
        "tuner": TunerStep,
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

        # Extract step-specific kwargs
        kwargs = {
            k: v
            for k, v in step_config.items()
            if k not in ["id", "type", "enabled", "depends_on"]
        }

        return step_class(
            step_id=step_id,
            cfg=cfg,
            enabled=enabled,
            depends_on=depends_on,
            **kwargs,
        )
