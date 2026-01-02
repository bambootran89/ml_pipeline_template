"""Model evaluation pipeline step."""

from __future__ import annotations

from typing import Any, Dict

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
    in the experiment.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Any = None,
        model_step_id: str = "train_model",
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
        """
        super().__init__(step_id, cfg, enabled, depends_on)
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

        # Get model and data from context
        model_key = f"{self.model_step_id}_model"
        dm_key = f"{self.model_step_id}_datamodule"

        if model_key not in context or dm_key not in context:
            raise ValueError(
                f"Step '{self.step_id}' requires " f"'{model_key}' and '{dm_key}'"
            )

        wrapper = context[model_key]
        dm = context[dm_key]

        # Build datamodule if None (from ModelLoaderStep)
        if dm is None:
            print(f"[{self.step_id}] Building datamodule from preprocessed_data")
            if "preprocessed_data" not in context:
                raise ValueError("Missing 'preprocessed_data' in context")

            df = context["preprocessed_data"]

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

        context[f"{self.step_id}_metrics"] = metrics

        print(f"[{self.step_id}] Evaluation complete")
        print(f"  Metrics: {metrics}")

        return context
