"""Model evaluation pipeline step with data wiring support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.eval.base import BaseEvaluator
from mlproject.src.eval.classification import ClassificationEvaluator
from mlproject.src.eval.clustering import ClusteringEvaluator
from mlproject.src.eval.regression import RegressionEvaluator
from mlproject.src.eval.timeseries import TimeSeriesEvaluator
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


def _ensure_numpy(x: Any) -> np.ndarray:
    """Convert pandas DataFrame to numpy array if needed."""
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    return np.asarray(x, dtype=float)


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
        """Evaluate model on provided predictions
        and targets or fallback to datamodule."""
        self.validate_dependencies(context)
        # MODE 1: Evaluate using provided predictions
        p = self.get_input(context, "predictions", required=False)
        t = self.get_input(context, "targets", required=False)
        if t is not None and p is not None:
            y_true = _ensure_numpy(t)
            y_pred = _ensure_numpy(p)

            metrics = self.evaluator.evaluate(y_true, y_pred)
            self.set_output(context, "metrics", metrics)

            print(f"[{self.step_id}] Evaluation complete")
            print(f"  Metrics: {metrics}")
            return context
        # MODE 2: Run prediction via model wrapper
        mk = f"{self.model_step_id}_model"
        dk = f"{self.model_step_id}_datamodule"
        wrapper = context.get(mk)
        dm = context.get(dk)
        if wrapper is None:
            raise ValueError(
                f"Step '{self.step_id}': Model not found in context['{mk}']."
            )

        if dm is None:
            df = self._build_eval_frame(context)
            dm = DataModuleFactory.build(self.cfg, df)

        y_true, y_pred = self._eval_from_datamodule(wrapper, dm)
        metrics = self.evaluator.evaluate(y_true, y_pred, x=y_pred)

        self.set_output(context, "metrics", metrics)

        print(f"[{self.step_id}] Evaluation complete")
        print(f"  Metrics: {metrics}")
        return context

    def _build_eval_frame(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Assemble evaluation DataFrame from features and targets."""
        f = self.get_input(context, "features")
        tg = self.get_input(context, "targets", required=False)
        if f is None:
            raise ValueError(f"Step '{self.step_id}': features is None.")

        data_cfg: Dict[str, Any] = self.cfg.get("data", {})
        data_type: str = str(data_cfg.get("type", "tabular")).lower()

        if data_type == "timeseries":
            return f.copy()

        if tg is not None:
            tg = _ensure_numpy(tg)
            return pd.concat([f, pd.DataFrame(tg)], axis=1)

        return f.copy()

    def _eval_from_datamodule(
        self, wrapper: Any, dm: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run prediction and return true labels and predictions as numpy."""
        if hasattr(dm, "get_test_windows"):
            x, y = dm.get_test_windows()
            return y, wrapper.predict(x)

        data = dm.get_data()
        x_test, y_test = data[-2], data[-1]
        return y_test, wrapper.predict(x_test)


# Register step type
StepFactory.register("evaluator", EvaluatorStep)
