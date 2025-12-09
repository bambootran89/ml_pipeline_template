from typing import Any

import numpy as np
from xgboost import XGBRegressor

from mlproject.src.models.base import MLModelWrapper


class XGBWrapper(MLModelWrapper):
    """
    Wrapper for XGBoost regressor using MLModelWrapper as base.
    """

    def build(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        n_estimators = self.cfg.get("n_estimators", -1)
        max_depth = self.cfg.get("max_depth", -1)
        learning_rate = self.cfg.get("learning_rate", -1)
        objective = self.cfg.get("objective", "reg:squarederror")

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
        )

    def predict(self, x: Any, **kwargs: Any) -> Any:
        """
        Predict using the trained XGBRegressor.

        Args:
            x (Any): Input features.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model not built/trained yet.")

        x_arr = np.asarray(x, dtype=np.float32)

        shape = x_arr.shape
        assert len(shape) <= 3

        if len(shape) == 3:
            x_arr = x_arr.reshape(-1, shape[1] * shape[2])

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        preds = self.model.predict(x_arr)
        return np.asarray(preds)
