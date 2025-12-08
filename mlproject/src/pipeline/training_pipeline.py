from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.evaluator import mae, mse, smape
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.trainer import train_model


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
        """
        Train and evaluate one approach.

        Args:
            approach: Experiment approach config.
            data: Preprocessed DataFrame (already fitted and transformed).
        """
        df = data
        batch_size = int(approach.get("hyperparams", {}).get("batch_size", 16))
        num_workers = self.cfg.training.get("num_workers", 0)

        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup(batch_size=batch_size, num_workers=num_workers)
        train_loader, val_loader, _, _ = dm.get_loaders()

        wrapper = self._init_model(approach)

        wrapper = train_model(
            wrapper,
            train_loader,
            val_loader,
            approach.get("hyperparams", {}),
            device=self.cfg.training.get("device", "cpu"),
            save_dir=self.cfg.training.get(
                "artifacts_dir", "mlproject/artifacts/models"
            ),
        )

        # Evaluation
        x_test, y_test = dm.get_test_windows()
        self.evaluate(wrapper, x_test, y_test)

    def evaluate(self, wrapper, x_test, y_test):
        """
        Evaluate model on test set and print metrics.

        Args:
            wrapper: Trained model wrapper.
            x_test: Test input windows.
            y_test: Test target values.
        """
        preds = wrapper.predict(x_test)
        print(
            f"MAE={mae(y_test, preds):.6f}, "
            f"MSE={mse(y_test, preds):.6f}, "
            f"SMAPE={smape(y_test, preds):.6f}"
        )
