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


def _ensure_df(x: Any) -> pd.DataFrame:
    """Convert pandas DataFrame to numpy array if needed."""
    if isinstance(x, np.ndarray):
        return pd.DataFrame(x)
    return x


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
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
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
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.step_eval_type = kwargs.get("step_eval_type", "")
        if "cluster" in self.step_id:
            self.step_eval_type = "clustering"
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> BaseEvaluator:
        """
        Build evaluator based on config.

        Returns
        -------
        BaseEvaluator
            Appropriate evaluator instance.
        """
        if len(self.step_eval_type) == 0:
            eval_type = self.cfg.get("evaluation", {}).get("type", "regression")
        else:
            eval_type = self.step_eval_type

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
        wrapper = self.get_input(context, "model", required=False)
        dm = self.get_input(context, "datamodule", required=False)

        if dm is None:
            df = self._build_eval_frame(context)
            dm = DataModuleFactory.build(self.cfg, df)
            dm.setup()
        y_true, y_pred, x_test = self._eval_from_datamodule(wrapper, dm)
        metrics = self.evaluator.evaluate(y_true, y_pred, x=x_test, model=wrapper)

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
            tg = _ensure_df(tg)
            return pd.concat([f, tg], axis=1)

        return f.copy()

    def _eval_from_datamodule(
        self, wrapper: Any, dm: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run prediction and return true labels and predictions as numpy."""
        if hasattr(dm, "get_test_windows"):
            x, y = dm.get_test_windows()
            return y, wrapper.predict(x), x

        data = dm.get_data()
        x_test, y_test = data[-2], data[-1]
        return y_test, wrapper.predict(x_test), x_test


# Register step type
StepFactory.register("evaluator", EvaluatorStep)
