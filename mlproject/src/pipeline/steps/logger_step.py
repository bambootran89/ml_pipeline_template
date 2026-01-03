"""MLflow logging pipeline step with wiring support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow


class LoggerStep(BasePipelineStep):
    """
    Log artifacts and metrics to MLflow.

    Extracts outputs from previous steps and logs them
    to MLflow Model Registry. Supports data wiring for
    flexible input key mapping.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    <model_step_id>_model : ModelWrapper
        Trained model to log.
    preprocessor : OfflinePreprocessor
        Fitted preprocessor.
    <eval_step_id>_metrics : Dict
        Evaluation metrics.

    Wiring Example
    --------------
    ::

        - id: "log_results"
          type: "logger"
          depends_on: ["evaluate"]
          wiring:
            inputs:
              model: "ensemble_model"
              metrics: "final_metrics"
          model_step_id: "train_ensemble"
          eval_step_id: "evaluate"
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        model_step_id: str = "train_model",
        eval_step_id: str = "evaluate",
        **kwargs: Any,
    ) -> None:
        """
        Initialize MLflow logging step.

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
        model_step_id : str, default="train_model"
            Step ID that trained the model.
        eval_step_id : str, default="evaluate"
            Step ID that computed metrics.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.model_step_id = model_step_id
        self.eval_step_id = eval_step_id
        self.mlflow_manager = MLflowManager(cfg)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log components to MLflow.

        Uses wiring configuration for input key mapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Unchanged context.
        """
        self.validate_dependencies(context)

        if not self.mlflow_manager.enabled:
            print(f"[{self.step_id}] MLflow disabled, skipping")
            return context

        experiment_name = self.cfg.experiment.get("name", "undefined")

        # Get inputs using wiring or default patterns
        model_key = f"{self.model_step_id}_model"
        metrics_key = f"{self.eval_step_id}_metrics"

        wrapper = self.get_input(
            context, "model", default_key=model_key, required=False
        )
        preprocessor = self.get_input(
            context, "preprocessor", default_key="preprocessor", required=False
        )
        metrics = (
            self.get_input(context, "metrics", default_key=metrics_key, required=False)
            or {}
        )

        run_name = f"{experiment_name}_run"

        with self.mlflow_manager.start_run(run_name=run_name):
            # Log preprocessor
            if preprocessor is not None:
                self.mlflow_manager.log_component(
                    obj=preprocessor.transform_manager,
                    name=f"{experiment_name}_preprocessor",
                    artifact_type="preprocess",
                )

            # Log model
            if wrapper is not None:
                self.mlflow_manager.log_component(
                    obj=wrapper,
                    name=f"{experiment_name}_model",
                    artifact_type="model",
                )

            # Log metrics
            safe_metrics = flatten_metrics_for_mlflow(metrics)
            hyperparams = dict(self.cfg.experiment.get("hyperparams", {}))
            self.mlflow_manager.log_metadata(params=hyperparams, metrics=safe_metrics)

        print(f"[{self.step_id}] Logged to MLflow: {run_name}")

        return context


# Register step type
StepFactory.register("logger", LoggerStep)
