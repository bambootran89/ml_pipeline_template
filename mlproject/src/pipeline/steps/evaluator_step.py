"""Enhanced evaluator step with multi-source feature support."""

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
from mlproject.src.pipeline.steps.base import PipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


def _ensure_numpy(x: Any) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    return np.asarray(x, dtype=float)


class EvaluatorStep(PipelineStep):
    """Evaluate model with multi-source feature composition support.

    This step composes features from multiple sources before evaluation,
    allowing models trained on composed features to be evaluated properly.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    features : pd.DataFrame or array-like
        Primary feature source.
    additional_feature_keys : List[str], optional
        Additional feature source keys to compose.
    targets : pd.DataFrame or array-like, optional
        Target labels for evaluation.
    predictions : np.ndarray, optional
        Pre-computed predictions (MODE 1).
    model : Any, optional
        Model wrapper for inference (MODE 2).
    datamodule : Any, optional
        Pre-built DataModule (MODE 2).

    Context Outputs
    ---------------
    metrics : Dict[str, float]
        Evaluation metrics.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        step_eval_type: str = "",
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize enhanced evaluator.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : Any
            Configuration object.
        enabled : bool
            Whether step should execute.
        depends_on : Optional[List[str]]
            Prerequisite steps.
        step_eval_type : str
            Override evaluation type (clustering, classification, etc).
        additional_feature_keys : Optional[List[str]]
            Keys for additional feature sources.
        feature_align_method : str
            Method for aligning feature shapes.
        **kwargs
            Additional parameters.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )

        self.step_eval_type = step_eval_type
        if "cluster" in self.step_id:
            self.step_eval_type = "clustering"

        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> BaseEvaluator:
        """Build evaluator based on config.

        Returns
        -------
        BaseEvaluator
            Appropriate evaluator instance.
        """
        if self.step_eval_type:
            eval_type = self.step_eval_type
        else:
            eval_type = self.cfg.get("evaluation", {}).get("type", "regression")

        evaluator_map = {
            "classification": ClassificationEvaluator,
            "regression": RegressionEvaluator,
            "clustering": ClusteringEvaluator,
            "timeseries": TimeSeriesEvaluator,
        }

        evaluator_class = evaluator_map.get(eval_type)
        if evaluator_class is None:
            raise ValueError(f"Unsupported eval type: {eval_type}")

        return evaluator_class()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluation with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with metrics added.
        """
        self.validate_dependencies(context)

        # MODE 1: Evaluate using pre-computed predictions
        preds = self.get_input(context, "predictions", required=False)
        targets = self.get_input(context, "targets", required=False)

        if preds is not None and targets is not None:
            metrics = self._eval_from_predictions(preds, targets)
            self.set_output(context, "metrics", metrics)
            self._log_metrics(metrics)
            return context

        # MODE 2: Compose features and run model inference
        model = self.get_input(context, "model", required=False)
        datamodule = self.get_input(context, "datamodule", required=False)

        if datamodule is None:
            datamodule = self._build_datamodule_with_composition(context)

        y_true, y_pred, x_test = self._eval_from_datamodule(model, datamodule)

        metrics = self.evaluator.evaluate(y_true, y_pred, x=x_test, model=model)

        self.set_output(context, "metrics", metrics)
        self._log_metrics(metrics)

        return context

    def _eval_from_predictions(
        self,
        preds: Any,
        targets: Any,
    ) -> Dict[str, float]:
        """Evaluate from pre-computed predictions.

        Parameters
        ----------
        preds : Any
            Model predictions.
        targets : Any
            True labels.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics.
        """
        y_true = _ensure_numpy(targets)
        y_pred = _ensure_numpy(preds)

        return self.evaluator.evaluate(y_true, y_pred)

    def _build_datamodule_with_composition(
        self,
        context: Dict[str, Any],
    ) -> Any:
        """Build DataModule with composed features.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Any
            DataModule instance.
        """
        # Compose features
        features_df, _ = self.get_composed_features(context, "features", required=True)

        # Get targets
        targets_df = self.get_input(context, "targets", required=False)

        # Build input DataFrame
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", "tabular")).lower()

        if data_type == "timeseries":
            input_df = features_df.copy()
        elif targets_df is not None:
            input_df = pd.concat([features_df, targets_df], axis=1)
        else:
            input_df = features_df.copy()

        # Build DataModule
        print(
            f"[{self.step_id}] Building DataModule with composed features: "
            f"{input_df.shape}"
        )

        dm = DataModuleFactory.build(self.cfg, input_df)
        dm.setup()

        return dm

    def _eval_from_datamodule(
        self,
        model: Any,
        datamodule: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run prediction and get results.

        Parameters
        ----------
        model : Any
            Model wrapper.
        datamodule : Any
            DataModule instance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            True labels, predictions, test features.
        """
        if hasattr(datamodule, "get_test_windows"):
            x_test, y_true = datamodule.get_test_windows()
        else:
            data = datamodule.get_data()
            x_test, y_true = data[-2], data[-1]

        y_pred = model.predict(x_test)

        return y_true, y_pred, x_test

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics.

        Parameters
        ----------
        metrics : Dict[str, float]
            Evaluation metrics.
        """
        print(f"[{self.step_id}] Evaluation complete")
        print(f"  Metrics: {metrics}")


StepFactory.register("evaluator", EvaluatorStep)
