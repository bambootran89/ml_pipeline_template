"""Model evaluation pipeline step with data wiring support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification import ClassificationEvaluator
from mlproject.src.eval.clustering import ClusteringEvaluator
from mlproject.src.eval.regression import RegressionEvaluator
from mlproject.src.eval.timeseries import TimeSeriesEvaluator
from mlproject.src.pipeline.steps.base import BasePipelineStep


class EvaluatorStep(BasePipelineStep):
    """
    Evaluate trained model on test data.

    Computes metrics based on evaluation type configured
    in the experiment. Supports data wiring for flexible
    model/data source configuration.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    <model_step_id>_model : ModelWrapper
        Trained model to evaluate (default: train_model_model).
    <model_step_id>_datamodule : DataModule
        DataModule with test data (default: train_model_datamodule).
    preprocessed_data : pd.DataFrame
        Fallback if datamodule is None.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    <step_id>_metrics : Dict[str, float]
        Evaluation metrics.

    Wiring Example
    --------------
    ::

        - id: "eval_ensemble"
          type: "evaluator"
          depends_on: ["train_xgb", "train_catboost"]
          wiring:
            inputs:
              model: "ensemble_model"      # Custom model key
              datamodule: "shared_dm"      # Custom datamodule key
            outputs:
              metrics: "ensemble_metrics"  # Custom output key
          model_step_id: "train_xgb"       # Fallback if wiring not set
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        model_step_id: str = "train_model",
        **kwargs: Any,
    ) -> None:
        """
        Initialize evaluation step.

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
            ID of step that trained the model to evaluate.
            Used for default key patterns.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.model_step_id = model_step_id
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> BaseEvaluator:
        """
        Build evaluator based on config.

        Returns
        -------
        BaseEvaluator
            Appropriate evaluator instance.
        """
        eval_type = self.cfg.get("evaluation", {}).get("type", "regression")

        if eval_type == "classification":
            return ClassificationEvaluator()
        elif eval_type == "regression":
            return RegressionEvaluator()
        elif eval_type == "clustering":
            return ClusteringEvaluator()
        elif eval_type == "timeseries":
            return TimeSeriesEvaluator()
        else:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model and store metrics.

        Uses wiring configuration for input/output key mapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Must contain trained model and datamodule.

        Returns
        -------
        Dict[str, Any]
            Context with evaluation metrics added.
        """
        self.validate_dependencies(context)

        # Get model using wiring or default pattern
        model_key = f"{self.model_step_id}_model"
        dm_key = f"{self.model_step_id}_datamodule"

        wrapper = self.get_input(context, "model", default_key=model_key, required=True)
        dm = self.get_input(context, "datamodule", default_key=dm_key, required=False)

        # Build datamodule if None (from ModelLoaderStep)
        if dm is None:
            print(f"[{self.step_id}] Building datamodule from preprocessed_data")
            df = self.get_input(
                context, "data", default_key="preprocessed_data", required=True
            )
            dm = DataModuleFactory.build(self.cfg, df)

        # Get test data
        if hasattr(dm, "get_test_windows"):
            x_test, y_test = dm.get_test_windows()
        else:
            _, _, _, _, x_test, y_test = dm.get_data()

        # Predict and evaluate
        preds = wrapper.predict(x_test)
        metrics = self.evaluator.evaluate(
            y_test, preds, x=x_test, model=wrapper.get_model()
        )

        # Store output using wiring
        self.set_output(context, "metrics", metrics)

        print(f"[{self.step_id}] Evaluation complete")
        print(f"  Metrics: {metrics}")

        return context
