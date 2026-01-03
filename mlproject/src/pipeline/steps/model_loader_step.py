"""Model loading step with data wiring support.

This module loads pre-trained models from MLflow Model Registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager


class ModelLoaderStep(BasePipelineStep):
    """Load a pre-trained model from MLflow Model Registry.

    Supports data wiring for custom output key mapping.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    model : Any
        Loaded model wrapper from MLflow.
    datamodule : None
        Set to None - EvaluatorStep will build when needed.

    Wiring Example
    --------------
    ::

        - id: "load_model"
          type: "model_loader"
          alias: "production"
          wiring:
            outputs:
              model: "production_model"

    Configuration
    -------------
    experiment_name : str, optional
        Model name in registry (defaults to cfg.experiment.name).
    alias : str, optional
        Model version alias (default: "latest").
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        alias: str = "latest",
        **kwargs: Any,
    ) -> None:
        """Initialize model loader step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration object.
        enabled : bool, default=True
            Whether step is active.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        alias : str, optional
            MLflow registry alias (default: "latest").
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.alias = alias
        self.mlflow_manager = MLflowManager(self.cfg)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load model from MLflow Model Registry.

        Uses wiring configuration for output key mapping.

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

        experiment_name = self.cfg.experiment.get("name", "")
        if not experiment_name:
            raise ValueError("experiment.name must be specified in config")

        registry_name = f"{experiment_name}_model"

        print(
            f"[{self.step_id}] Loading model from MLflow: "
            f"name='{registry_name}', alias='{self.alias}'"
        )

        model = self.mlflow_manager.load_component(
            name=registry_name,
            alias=self.alias,
        )

        if model is None:
            raise RuntimeError(
                f"Failed to load model '{registry_name}' with alias "
                f"'{self.alias}'. Make sure model is registered in MLflow."
            )

        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Loaded model ({type(model)}) does not have 'predict' method."
            )

        # Store outputs using wiring
        self.set_output(context, "model", model)
        self.set_output(context, "datamodule", None)

        print(
            f"[{self.step_id}] Successfully loaded model: "
            f"{registry_name}@{self.alias}"
        )
        return context


# Register step type
StepFactory.register("model_loader", ModelLoaderStep)
