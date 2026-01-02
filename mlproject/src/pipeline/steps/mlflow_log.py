"""MLflow logging pipeline step."""

from __future__ import annotations

from typing import Any, Dict

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.func_utils import flatten_metrics_for_mlflow


class MLflowLogStep(BasePipelineStep):
    """
    Log artifacts and metrics to MLflow.

    Extracts outputs from previous steps and logs them
    to MLflow Model Registry.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Any = None,
        model_step_id: str = "train_model",
        eval_step_id: str = "evaluate",
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
        """
        super().__init__(step_id, cfg, enabled, depends_on)
        self.model_step_id = model_step_id
        self.eval_step_id = eval_step_id
        self.mlflow_manager = MLflowManager(cfg)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log components to MLflow.

        Parameters
        ----------
        context : Dict[str, Any]
            Must contain model, preprocessor, and metrics.

        Returns
        -------
        Dict[str, Any]
            Unchanged context.
        """
        self.validate_dependencies(context)

        if not self.mlflow_manager.enabled:
            print(f"[{self.step_id}] MLflow disabled, skipping")
            return context

        model_name = self.cfg.experiment.get("model", "model")

        # Get components from context
        wrapper = context.get(f"{self.model_step_id}_model")
        preprocessor = context.get("preprocessor")
        metrics = context.get(f"{self.eval_step_id}_metrics", {})

        run_name = f"{model_name}_run"

        with self.mlflow_manager.start_run(run_name=run_name):
            # Log preprocessor
            if preprocessor is not None:
                self.mlflow_manager.log_component(
                    obj=preprocessor.transform_manager,
                    name=f"{model_name}_preprocessor",
                    artifact_type="preprocess",
                )

            # Log model
            if wrapper is not None:
                self.mlflow_manager.log_component(
                    obj=wrapper,
                    name=f"{model_name}_model",
                    artifact_type="model",
                )

            # Log metrics
            safe_metrics = flatten_metrics_for_mlflow(metrics)
            hyperparams = dict(self.cfg.experiment.get("hyperparams", {}))
            self.mlflow_manager.log_metadata(params=hyperparams, metrics=safe_metrics)

        print(f"[{self.step_id}] Logged to MLflow: {run_name}")

        return context
