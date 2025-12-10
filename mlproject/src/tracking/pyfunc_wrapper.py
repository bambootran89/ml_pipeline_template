"""
PyFunc model wrapper for MLflow.

Provides MLflowModelWrapper to serve arbitrary model wrappers
through MLflow PyFunc interface, ensuring consistent input preprocessing.
"""

from typing import Any, Union

import mlflow.pyfunc
import numpy as np
import pandas as pd


class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
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
        self, context: Any, model_input: Union[np.ndarray, pd.DataFrame]
    ) -> Any:
        """
        Execute model prediction using standardized input.

        Converts pandas DataFrame inputs to numpy arrays of dtype float32,
        then forwards to the wrapped model's predict method.

        Args:
            context (Any): MLflow context (unused).
            model_input (Union[np.ndarray, pd.DataFrame]):
              Raw input data for prediction.

        Returns:
            Any: Predictions from the wrapped model.
        """
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values

        model_input = np.asarray(model_input, dtype=np.float32)
        return self.model_wrapper.predict(model_input)
