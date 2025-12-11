from typing import Any, Dict

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.trainer.trainer_factory import TrainerFactory


class FoldModelBuilder:
    """
    Factory helper for creating a model wrapper and its corresponding
    trainer for a single cross-validation fold.

    Centralizes model and trainer initialization so FoldRunner remains
    clean and independent of specific model/trainer creation logic.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def create(self, model_name: str, hyperparams: Dict[str, Any]):
        """
        Instantiate the model wrapper and trainer for the given model.

        Parameters
        ----------
        model_name : str
            Name of the model to create, as registered in ModelFactory.
        hyperparams : dict
            Dictionary of hyperparameters passed to the model constructor.

        Returns
        -------
        tuple
            (wrapper, trainer) where:
                - wrapper: model wrapper instance
                - trainer: trainer responsible for training the model
        """
        wrapper = ModelFactory.create(model_name, hyperparams)
        trainer = TrainerFactory.create(
            model_name, wrapper, self.cfg.training.artifacts_dir
        )
        return wrapper, trainer
