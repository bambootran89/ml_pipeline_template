from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.trainer_factory import TrainerFactory


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline for both deep learning and ML models.

    Responsibilities:
    - Preprocess data using OfflinePreprocessor
    - Build correct DataModule (DL or ML)
    - Initialize model via ModelFactory
    - Train model using TrainerFactory (DL or ML trainer)
    - Evaluate predictions with TimeSeriesEvaluator
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self):
        """Fit scaler once and return transformed dataset."""
        preprocessor = OfflinePreprocessor(self.cfg)
        return preprocessor.run()

    def _init_model(self, approach: Dict[str, Any]):
        """Initialize model wrapper from model factory."""
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.create(name, hp)

    def _run_dl(
        self, trainer, dm: TSDLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run deep learning training + evaluation."""
        train_loader, val_loader, _, _ = dm.get_loaders()

        wrapper = trainer.train(train_loader, val_loader, hyperparams)

        x_test, y_test = dm.get_test_windows()
        preds = wrapper.predict(x_test)

        return y_test, preds

    def _run_ml(
        self, trainer, dm: TSMLDataModule, wrapper, hyperparams: Dict[str, Any]
    ):
        """Run ML model (sklearn/xgboost/lightgbm) training + evaluation."""
        x_train, y_train, x_val, y_val, x_test, y_test = dm.get_data()

        wrapper = trainer.train(
            (x_train, y_train),
            (x_val, y_val),
            hyperparams,
        )

        preds = wrapper.predict(x_test)
        return y_test, preds

    def run_approach(self, approach: Dict[str, Any], data):
        """
        Train + evaluate one approach.
        No if/else on model type — only dispatch based on DataModule class.
        """

        df = data
        model_name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})

        wrapper = self._init_model(approach)
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        trainer = TrainerFactory.create(
            model_name, wrapper, self.cfg.training.artifacts_dir
        )

        # Dispatch theo DataModule class → không đụng vào model logic
        if isinstance(dm, TSDLDataModule):
            y_test, preds = self._run_dl(trainer, dm, wrapper, hyperparams)

        elif isinstance(dm, TSMLDataModule):
            y_test, preds = self._run_ml(trainer, dm, wrapper, hyperparams)

        else:
            raise NotImplementedError(f"Unsupported DataModule type: {type(dm)}")

        metrics = TimeSeriesEvaluator().evaluate(y_test, preds)
        print(metrics)
