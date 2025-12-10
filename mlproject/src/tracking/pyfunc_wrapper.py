"""
PyFunc model wrapper for MLflow.

Provides MLflowModelWrapper, enabling MLflow pyfunc serving for
custom model objects.
"""

import mlflow.pyfunc
import numpy as np
import pandas as pd


class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for serving arbitrary model wrappers.

    Ensures model_input is converted to float32 numpy array before
    being passed to underlying model.predict().
    """

    def __init__(self, model_wrapper):
        """
        Args:
            model_wrapper: Object implementing predict().
        """
        self.model_wrapper = model_wrapper

    def predict(self, context, model_input):
        """
        Execute model prediction.

        Args:
            context: MLflow context (unused).
            model_input: Raw input array or DataFrame.

        Returns:
            Predictions from the wrapped model.
        """
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values

        model_input = np.asarray(model_input, dtype=np.float32)
        return self.model_wrapper.predict(model_input)
