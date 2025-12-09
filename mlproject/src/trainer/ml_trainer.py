from typing import Any, Dict, Optional

from mlproject.src.trainer.base_trainer import BaseTrainer


class MLTrainer(BaseTrainer):
    """Trainer for non-deep-learning (classic ML) models."""

    def train(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        hyperparams: Optional[Dict] = None,
    ):
        """
        Train the model wrapper on provided datasets.

        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            hyperparams: Optional hyperparameters
        """
        _ = val_data  # mark as intentionally unused
        _ = hyperparams  # mark as intentionally unused
        x_train, y_train = train_data
        fit_kwargs = {}
        if val_data is not None:
            x_val, y_val = val_data
            fit_kwargs["x_val"] = x_val
            fit_kwargs["y_val"] = y_val

        self.wrapper.fit(x_train, y_train, **fit_kwargs)
        self.save()
        return self.wrapper
