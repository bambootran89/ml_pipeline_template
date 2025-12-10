from typing import Any

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class MLflowModelWrapper(PythonModel):
    """
    MLflow PyFunc wrapper for serving arbitrary model wrappers.

    Converts inputs to float32 numpy arrays before passing to the underlying model.
    Suitable for any model implementing a predict() method.

    Attributes:
        model_wrapper (Any): Model object implementing predict().
    """

    def __init__(self, model_wrapper: Any):
        """
        Initialize the PyFunc wrapper.

        Args:
            model_wrapper (Any): Object implementing predict().
        """
        self.model_wrapper = model_wrapper

    def predict(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = None
    ) -> Any:
        """
        Execute model prediction using standardized input.

        Converts pandas DataFrame inputs to numpy arrays of dtype float32,
        then forwards to the wrapped model's predict method.

        Args:
            context (Any): MLflow context (unused).
            model_input (Any): Raw input data for prediction.
            params (dict[str, Any] | None): Optional parameters (unused).

        Returns:
            Any: Predictions from the wrapped model.
        """
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values

        model_input = np.asarray(model_input, dtype=np.float32)
        return self.model_wrapper.predict(model_input)

    def predict_stream(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = None
    ) -> Any:
        """
        Implement abstract method predict_stream to satisfy PythonModel.

        Simply calls self.predict; MLflow streaming input is passed as model_input.

        Args:
            context (Any): MLflow context (unused)
            model_input (Any): Streaming input
            params (dict[str, Any] | None): Optional parameters (unused)
        """
        return self.predict(context, model_input, params)
