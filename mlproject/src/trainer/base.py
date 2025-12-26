import os
from typing import Any


class BaseTrainer:
    """Base trainer for both deep learning and classical ML models."""

    def __init__(self, wrapper: Any, save_dir: str = "mlproject/artifacts/models"):
        """
        Args:
            wrapper: Model wrapper (PyTorch or other)
            save_dir: Directory to save the trained model
        """
        self.wrapper = wrapper
        self.save_dir = save_dir

    def save(self):
        """Save the model or wrapper."""
        os.makedirs(self.save_dir, exist_ok=True)
        self.wrapper.save(self.save_dir)

    def train(self, datamodule: Any, hyperparams: dict[str, Any]) -> Any:
        """
        Abstract method to standardize training interface.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
