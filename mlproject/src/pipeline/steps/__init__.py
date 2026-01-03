"""Pipeline steps package.

This package provides all pipeline step implementations.
Importing this package triggers registration of all step types.

Import Order:
1. base.py - BasePipelineStep (no dependencies)
2. factory.py - StepFactory (empty registry)
3. All step files - each imports factory and registers itself
"""

from mlproject.src.pipeline.steps.advanced_step import (
    BranchStep,
    ParallelStep,
    SubPipelineStep,
)

# Base class (no circular import issues)
from mlproject.src.pipeline.steps.base import BasePipelineStep

# Import all steps to trigger registration
# Each step file imports StepFactory and calls register() at module load
from mlproject.src.pipeline.steps.data_loader_step import DataLoaderStep
from mlproject.src.pipeline.steps.evaluator_step import EvaluatorStep

# Factory (defines empty registry)
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.pipeline.steps.generic_model_step import (
    ClusteringModelStep,
    GenericModelStep,
)
from mlproject.src.pipeline.steps.inference_step import InferenceStep
from mlproject.src.pipeline.steps.logger_step import LoggerStep
from mlproject.src.pipeline.steps.model_loader_step import ModelLoaderStep
from mlproject.src.pipeline.steps.preprocessor_step import PreprocessorStep
from mlproject.src.pipeline.steps.trainer_step import TrainerStep
from mlproject.src.pipeline.steps.tuner_step import TunerStep

__all__ = [
    # Base
    "BasePipelineStep",
    # Factory
    "StepFactory",
    # Basic steps
    "DataLoaderStep",
    "PreprocessorStep",
    "TrainerStep",
    "EvaluatorStep",
    "InferenceStep",
    "ModelLoaderStep",
    "LoggerStep",
    "TunerStep",
    # Generic model steps
    "GenericModelStep",
    "ClusteringModelStep",
    # Advanced steps
    "ParallelStep",
    "BranchStep",
    "SubPipelineStep",
]
