from mlproject.src.trainer.dl_trainer import DeepLearningTrainer
from mlproject.src.trainer.ml_trainer import MLTrainer


class TrainerFactory:
    """
    Factory that returns the correct Trainer (DL or ML)
    depending on the model type.
    """

    DL_MODELS = {"tft", "nlinear", "lstm", "gru", "transformer"}
    ML_MODELS = {"xgboost", "xgb", "lgbm", "lightgbm", "rf", "svm", "sklearn"}

    @classmethod
    def create(cls, model_name: str, wrapper):
        """
        Create and return a Trainer instance for the given model type.

        Args:
            model_name (str): Name of model (e.g., 'tft', 'xgboost').
            wrapper: The model wrapper instance.

        Returns:
            DeepLearningTrainer | MLTrainer: Trainer suitable for the model.

        Raises:
            RuntimeError: If no trainer exists for the provided model name.
        """
        if model_name in cls.DL_MODELS:
            return DeepLearningTrainer(wrapper)
        if model_name in cls.ML_MODELS:
            return MLTrainer(wrapper)

        raise RuntimeError(f"No trainer available for model {model_name}")
