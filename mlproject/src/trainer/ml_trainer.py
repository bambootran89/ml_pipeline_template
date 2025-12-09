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
        self.wrapper.fit(train_data[0], train_data[1])
        self.save()
        return self.wrapper
