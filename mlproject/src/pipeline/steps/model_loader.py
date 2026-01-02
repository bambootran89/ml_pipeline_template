"""Model loading step for evaluation and test pipelines.

This module provides a step that loads pre-trained models from disk
and builds datamodule from preprocessed data for evaluation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.pipeline.steps.base import BasePipelineStep


class ModelLoaderStep(BasePipelineStep):
    """Load a pre-trained model from disk and build datamodule.

    This step loads a saved model without training and builds a datamodule
    from preprocessed data, used in eval/test pipelines.

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Preprocessed data (needed to build datamodule).

    Context Outputs
    ---------------
    <step_id>_model : Any
        Loaded model wrapper instance.
    <step_id>_datamodule : DataModule
        Built from preprocessed_data for evaluation.

    Configuration Parameters
    ------------------------
    model_path : str, required
        Path to saved model file (.pkl, .joblib, .pth, etc.).

    Examples
    --------
    YAML configuration::

        steps:
          - id: "load_model"
            type: "model_loader"
            enabled: true
            model_path: "artifacts/models/xgboost_best.pkl"
    """

    def __init__(self, *args, model_path: str = None, **kwargs) -> None:
        """Initialize model loader step.

        Parameters
        ----------
        model_path : str, optional
            Path to model file. Required for execution.
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load model from disk and build datamodule.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context. Must contain 'preprocessed_data'.

        Returns
        -------
        Dict[str, Any]
            Context with loaded model and built datamodule added under
            keys '<step_id>_model' and '<step_id>_datamodule'.

        Raises
        ------
        FileNotFoundError
            If model_path not specified or file doesn't exist.
        ValueError
            If preprocessed_data not in context.
        NotImplementedError
            If model file format not supported.
        """
        # Validate model path
        if self.model_path is None:
            raise FileNotFoundError(
                f"Step '{self.step_id}' requires 'model_path' parameter"
            )

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Validate preprocessed data is available
        if "preprocessed_data" not in context:
            raise ValueError(
                f"Step '{self.step_id}' requires 'preprocessed_data' in context. "
                f"Make sure PreprocessingStep runs before ModelLoaderStep."
            )

        # Load model based on file extension
        if model_file.suffix in [".pkl", ".pickle"]:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
        elif model_file.suffix == ".joblib":
            import joblib

            model = joblib.load(model_file)
        else:
            raise NotImplementedError(
                f"Model format '{model_file.suffix}' not supported yet. "
                f"Supported: .pkl, .pickle, .joblib"
            )

        # Handle dict-wrapped models (from some trainers)
        if isinstance(model, dict):
            # Try common keys
            if "model" in model:
                model = model["model"]
            elif "wrapper" in model:
                model = model["wrapper"]
            else:
                raise ValueError(
                    f"Loaded dict does not contain 'model' or 'wrapper' key. "
                    f"Available keys: {list(model.keys())}"
                )

        # Verify model has predict method
        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Loaded model ({type(model)}) does not have 'predict' method. "
                f"Make sure the saved file contains a model wrapper."
            )

        # Store model in context
        context[f"{self.step_id}_model"] = model

        # Set datamodule to None - EvaluationStep will build it when needed
        # (This avoids model_registry error in DataModuleFactory.build)
        context[f"{self.step_id}_datamodule"] = None

        print(
            f"[{self.step_id}] Loaded model from: {self.model_path} "
            f"(size: {model_file.stat().st_size / 1024:.1f} KB)"
        )
        return context
