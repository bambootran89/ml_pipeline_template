from typing import Any, Optional, cast

import numpy as np
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

from mlproject.src.models.base import MLModelWrapper
from mlproject.src.utils.shape_utils import flatten_timeseries


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
        early_stopping_rounds = self.cfg.get("early_stopping_rounds", 10)
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            early_stopping_rounds=early_stopping_rounds,
        )

    def fit(
        self,
        x,
        y,
        sample_weight: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Train model with sklearn-style estimator."""
        if self.model is None:
            if x.ndim not in [2, 3]:
                raise ValueError("Input features 'x' must be 2D or 3D numpy arrays.")
            input_dim: int
            if x.ndim == 2:
                input_dim = x.shape[1]
            else:  # x.ndim == 3
                input_dim = x.shape[1] * x.shape[2]

            # Output dimension (1D -> 1, 2D -> features)
            output_dim: int = y.shape[-1] if y.ndim > 1 else 1

            self.build(input_dim, output_dim)  # Dùng giá trị int đã tính toán

        self.ensure_built()

        x_reshaped = flatten_timeseries(x)

        # Tạo eval_set nếu có validation data
        fit_params = kwargs.copy()
        if x_val is not None and y_val is not None:
            x_val_reshaped = flatten_timeseries(x_val)
            fit_params["eval_set"] = [(x_val_reshaped, y_val)]
            fit_params["verbose"] = False
        model = cast(BaseEstimator, self.model)
        model.fit(x_reshaped, y, sample_weight, **fit_params)

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
