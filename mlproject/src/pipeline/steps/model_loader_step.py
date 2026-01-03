"""Model loading step for evaluation and test pipelines.

This module loads pre-trained models from MLflow Model Registry
instead of local files for consistency.
"""

from __future__ import annotations

from typing import Any, Dict

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.tracking.mlflow_manager import MLflowManager


class ModelLoaderStep(BasePipelineStep):
    """Load a pre-trained model from MLflow Model Registry.

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Preprocessed data (for reference).

    Context Outputs
    ---------------
    <step_id>_model : Any
        Loaded model wrapper from MLflow.
    <step_id>_datamodule : None
        Set to None - EvaluationStep will build when needed.

    Configuration Parameters
    ------------------------
    experiment_name : str, optional
        Model name in registry (defaults to cfg.experiment.name).
    alias : str, optional
        Model version alias (default: "latest").

    Examples
    --------
    YAML configuration::

        steps:
          - id: "load_model"
            type: "model_loader"
            enabled: true
            alias: "production"  # or "latest", "staging"
    """

    def __init__(self, *args, alias: str = "latest", **kwargs) -> None:
        """Initialize model loader step.

        Parameters
        ----------
        alias : str, optional
            MLflow registry alias (default: "latest").
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.alias = alias
        self.mlflow_manager = MLflowManager(self.cfg)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load model from MLflow Model Registry.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with loaded model added.

        Raises
        ------
        RuntimeError
            If MLflow is disabled or model not found.
        """
        if not self.mlflow_manager.enabled:
            raise RuntimeError(
                f"Step '{self.step_id}' requires MLflow to be enabled. "
                f"Set mlflow.enabled=true in config."
            )

        # Get model name from config
        experiment_name = self.cfg.experiment.get("name", "").lower()
        if not experiment_name:
            raise ValueError("experiment.name must be specified in config")

        registry_name = f"{experiment_name}_model"

        print(
            f"[{self.step_id}] Loading model from MLflow: "
            f"name='{registry_name}', alias='{self.alias}'"
        )

        # Load model from MLflow
        model = self.mlflow_manager.load_component(
            name=registry_name,
            alias=self.alias,
        )

        if model is None:
            raise RuntimeError(
                f"Failed to load model '{registry_name}' with alias '{self.alias}'. "
                f"Make sure model is registered in MLflow."
            )

        # Verify model has predict method
        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Loaded model ({type(model)}) does not have 'predict' method."
            )

        # Store model in context
        context[f"{self.step_id}_model"] = model

        # Set datamodule to None - EvaluationStep will build when needed
        context[f"{self.step_id}_datamodule"] = None

        print(
            f"[{self.step_id}] Successfully loaded model: "
            f"{registry_name}@{self.alias}"
        )
        return context
