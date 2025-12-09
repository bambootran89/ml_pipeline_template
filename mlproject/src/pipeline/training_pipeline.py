from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.dl_trainer import DeepLearningTrainer


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline for time-series forecasting.

    Responsibilities:
    - Fit preprocessing (scaler) on train data ONCE
    - Transform data using fitted preprocessing
    - Train model
    - Evaluate model
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self):
        """
        Fit preprocessing pipeline and transform data.

        Returns:
            pd.DataFrame: Transformed dataset with fitted scaler saved.
        """
        preprocessor = OfflinePreprocessor(self.cfg)
        df = preprocessor.run()  # Fits scaler internally
        return df

    def _init_model(self, approach: Dict[str, Any]):
        """
        Initialize model wrapper from approach config.

        Args:
            approach: Experiment approach configuration.

        Returns:
            Model wrapper instance (NLinearWrapper or TFTWrapper).
        """
        name = approach["model"]
        hp = approach.get("hyperparams", {})

        # Convert DictConfig → dict
        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        if name == "nlinear":
            return NLinearWrapper(hp)
        if name == "tft":
            return TFTWrapper(hp)

        raise RuntimeError(f"Unknown model {name}")

    def run_approach(self, approach: Dict[str, Any], data):
        """Train and evaluate one approach."""
        df = data
        batch_size = int(approach.get("hyperparams", {}).get("batch_size", 16))
        num_workers = self.cfg.training.get("num_workers", 0)

        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup(batch_size=batch_size, num_workers=num_workers)
        train_loader, val_loader, _, _ = dm.get_loaders()

        wrapper = self._init_model(approach)

        # Use DeepLearningTrainer for PyTorch wrappers
        trainer = DeepLearningTrainer(
            wrapper, device=self.cfg.training.get("device", "cpu")
        )
        wrapper = trainer.train(
            train_loader, val_loader, approach.get("hyperparams", {})
        )

        # Evaluation
        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)
        metrics = TimeSeriesEvaluator().evaluate(y_test, preds)

        print(metrics)
