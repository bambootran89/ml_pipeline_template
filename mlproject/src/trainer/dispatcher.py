from mlproject.src.trainer.dl_trainer import DeepLearningTrainer

from ..datamodule.loader_factory import FoldDataLoaderBuilder


class FoldTrainer:
    """
    Fold-level training helper that selects the appropriate training
    procedure depending on whether the model is a deep learning model
    or a classical machine learning model.

    This class abstracts away training differences so FoldRunner
    can call a unified interface regardless of model type.
    """

    @staticmethod
    def train(wrapper, trainer, hyperparams, train_data, test_data):
        """
        Train a model using the correct training strategy.

        Parameters
        ----------
        wrapper : Any
            Model wrapper instance.
        trainer : Any
            Trainer instance (DL or ML).
        hyperparams : dict
            Hyperparameters for training.
        train_data : tuple
            (x_train, y_train) arrays.
        test_data : tuple
            (x_test, y_test) arrays.

        Returns
        -------
        Any
            The trained model wrapper.
        """
        if isinstance(trainer, DeepLearningTrainer):
            return FoldTrainer._train_dl(
                wrapper, trainer, hyperparams, train_data, test_data
            )
        return trainer.train(train_data, test_data, hyperparams)

    @staticmethod
    def _train_dl(wrapper, trainer, hyperparams, train_data, test_data):
        """
        Deep learning training workflow for a CV fold.

        Builds the model, constructs dataloaders, and runs the trainer.

        Parameters
        ----------
        wrapper : Any
            DL model wrapper with a `build()` method.
        trainer : DeepLearningTrainer
            Trainer handling forward/backward passes.
        hyperparams : dict
            Training hyperparameters.
        train_data : tuple
            (x_train, y_train).
        test_data : tuple
            (x_test, y_test).

        Returns
        -------
        Any
            Trained wrapper instance.
        """
        x_train, y_train = train_data
        x_test, y_test = test_data

        # Build model with correct input/output dims
        wrapper.build(
            input_dim=x_train.shape[1] * x_train.shape[2],
            output_dim=y_train.shape[1],
        )

        batch_size = int(hyperparams.get("batch_size", 16))
        train_loader, val_loader = FoldDataLoaderBuilder.build(
            x_train, y_train, x_test, y_test, batch_size
        )

        return trainer.train(train_loader, val_loader, hyperparams)
