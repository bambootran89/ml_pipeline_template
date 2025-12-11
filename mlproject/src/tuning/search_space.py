"""
Search space definitions for supported models.

Each model defines:
- Parameter ranges
- Sampling strategy (log, uniform, categorical)
- Optional step size (for integer parameters)
- Optional logarithmic sampling (for float parameters)
"""

from typing import Any, Dict


class SearchSpaceRegistry:
    """
    Registry that maintains hyperparameter search spaces for each model.

    Search space format example:
        {
            "param_name": {
                "type": "int" | "float" | "categorical",
                "range": [min, max] or [choices],
                "log": True/False,    # for float parameters
                "step": int           # for integer parameters (optional)
            }
        }

    This registry allows:
    - Fetching predefined search spaces
    - Registering custom model search spaces
    - Listing all available model definitions
    """

    SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
        "xgboost": {
            "n_estimators": {"type": "int", "range": [50, 500], "step": 10},
            "max_depth": {"type": "int", "range": [3, 10], "step": 1},
            "learning_rate": {
                "type": "float",
                "range": [0.001, 0.3],
                "log": True,
            },
            "subsample": {"type": "float", "range": [0.5, 1.0], "log": False},
            "colsample_bytree": {
                "type": "float",
                "range": [0.5, 1.0],
                "log": False,
            },
            "min_child_weight": {"type": "int", "range": [1, 10], "step": 1},
            "gamma": {"type": "float", "range": [0.0, 1.0], "log": False},
        },
        "nlinear": {
            "hidden": {"type": "int", "range": [64, 512], "step": 64},
            "lr": {
                "type": "float",
                "range": [1e-4, 1e-2],
                "log": True,
            },
            "n_epochs": {"type": "int", "range": [5, 30], "step": 5},
            "batch_size": {
                "type": "categorical",
                "range": [16, 32, 64],
            },
        },
        "tft": {
            "hidden_size": {"type": "int", "range": [32, 256], "step": 32},
            "num_layers": {"type": "int", "range": [1, 3], "step": 1},
            "lr": {
                "type": "float",
                "range": [1e-4, 1e-2],
                "log": True,
            },
            "n_epochs": {"type": "int", "range": [5, 30], "step": 5},
            "batch_size": {
                "type": "categorical",
                "range": [16, 32, 64],
            },
            "dropout": {"type": "float", "range": [0.0, 0.3], "log": False},
        },
    }

    @classmethod
    def get(cls, model_name: str) -> Dict[str, Any]:
        """
        Retrieve the search space for a given model.

        Args:
            model_name: Model name in lowercase.

        Returns:
            dict: Search space definition for the model.

        Raises:
            ValueError: If no search space exists for the given model.
        """
        model_name = model_name.lower()

        if model_name not in cls.SEARCH_SPACES:
            available = list(cls.SEARCH_SPACES.keys())
            raise ValueError(
                f"No search space defined for '{model_name}'. "
                f"Available: {available}"
            )

        return cls.SEARCH_SPACES[model_name]

    @classmethod
    def register(cls, model_name: str, search_space: Dict[str, Any]) -> None:
        """
        Register a new or custom search space for a model.

        Args:
            model_name: Name of the model.
            search_space: Dictionary describing the parameter space.
        """
        cls.SEARCH_SPACES[model_name.lower()] = search_space

    @classmethod
    def list_models(cls) -> list:
        """
        Return the list of supported models.

        Returns:
            list: Model names available in the registry.
        """
        return list(cls.SEARCH_SPACES.keys())
