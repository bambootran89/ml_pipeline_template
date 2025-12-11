from typing import Any, Dict, Optional

from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.models.xgboost_wrapper import XGBWrapper
from mlproject.src.utils.factory_base import DynamicFactoryBase

class ModelFactory(DynamicFactoryBase):
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
    def create(cls, name: str, hp: Dict[str, Any], model_registry: Optional[Dict] = None,):
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

        if model_registry is None: 
            pass 
        entry = model_registry.get(name)

        if not entry:
            raise ValueError(f"Model '{name}' not found in registry.")
        
        model_class = cls._get_class_from_config(entry)
        return model_class(cfg=hp)
    
        # name = name.lower()
        # if name not in cls.REGISTRY:
        #     raise RuntimeError(
        #         f"Unknown model '{name}'. Supported: {list(cls.REGISTRY)}"
        #     )
        # return cls.REGISTRY[name](hp)

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
