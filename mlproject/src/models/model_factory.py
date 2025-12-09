from typing import Any, Dict

from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.models.xgboost_wrapper import XGBWrapper


class ModelFactory:
    """
    Factory for creating and loading model wrappers.

    This class maps a model name to its corresponding wrapper class.
    It supports dynamic creation and loading of model instances
    without using if/else logic.
    """

    REGISTRY = {
        "nlinear": NLinearWrapper,
        "tft": TFTWrapper,
        "xgboost": XGBWrapper,
    }

    @classmethod
    def create(cls, name: str, hp: Dict[str, Any]):
        """
        Create a model wrapper instance.

        Args:
            name: Model name (string key in registry).
            hp: Hyperparameters dictionary passed to the wrapper.

        Returns:
            Instantiated model wrapper.

        Raises:
            RuntimeError: If model name is not registered.
        """
        name = name.lower()
        if name not in cls.REGISTRY:
            raise RuntimeError(
                f"Unknown model '{name}'. Supported: {list(cls.REGISTRY)}"
            )
        return cls.REGISTRY[name](hp)

    @classmethod
    def load(cls, name: str, hyperparams: Dict[str, Any], artifact_dir: str):
        """
        Load a trained model wrapper from artifacts directory.

        Args:
            name: Model name.
            hyperparams: Model hyperparameters.
            artifact_dir: Directory containing saved model files.

        Returns:
            Loaded model wrapper instance.

        Raises:
            RuntimeError: If model name is not registered.
        """
        name = name.lower()
        if name not in cls.REGISTRY:
            raise RuntimeError(
                f"Unknown model '{name}'. Supported: {list(cls.REGISTRY)}"
            )

        wrapper_cls = cls.REGISTRY[name]
        wrapper = wrapper_cls(hyperparams)
        wrapper.load(artifact_dir)
        return wrapper
