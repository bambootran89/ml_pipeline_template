from typing import Any, Tuple

from mlproject.src.eval.classification_eval import ClassificationEvaluator
from mlproject.src.eval.regression_eval import RegressionEvaluator
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator


class FoldEvaluator:
    """
    Wrapper around TimeSeriesEvaluator to compute metrics for a single
    cross-validation fold. Keeps evaluation logic isolated from FoldRunner.
    """

    def __init__(self, cfg):
        assert cfg is not None
        eval_type = cfg.get("evaluation", {}).get("type", "regression")
        if eval_type == "classification":
            self.evaluator = ClassificationEvaluator()
        elif eval_type == "regression":
            self.evaluator = RegressionEvaluator()
        else:
            self.evaluator = TimeSeriesEvaluator()

    def evaluate(self, wrapper: Any, test_data: Tuple):
        """
        Run model prediction and compute evaluation metrics on test data.

        Parameters
        ----------
        wrapper : Any
            Model wrapper providing a `predict(x)` method.
        test_data : tuple
            (x_test, y_test) arrays for this fold.

        Returns
        -------
        dict[str, float]
            Dictionary of evaluation metrics converted to Python floats.
        """
        x_test, y_test = test_data
        preds = wrapper.predict(x_test)
        metrics = self.evaluator.evaluate(y_test, preds)
        return {k: float(v) for k, v in metrics.items()}
