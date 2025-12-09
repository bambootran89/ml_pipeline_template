from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule
from mlproject.src.eval.ts_eval import TimeSeriesEvaluator
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor


class EvalPipeline(BasePipeline):
    """
    Evaluation-only pipeline.

    Responsibilities:
    - Load preprocessed dataset (or transform using SAVED scaler)
    - Load trained model weights
    - Evaluate model

    IMPORTANT: Does NOT fit scaler, only transforms using saved scaler.
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self):
        """
        Transform data using SAVED scaler (does NOT fit).

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        preprocessor = OfflinePreprocessor(self.cfg)

        df = preprocessor.load_raw_data()
        df = preprocessor.transform(df)
        return df

    def _load_model(self, approach):
        """
        Load trained model wrapper from artifacts.
        """
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def run_approach(self, approach, data):
        """
        Evaluate a single approach.
        """
        df = data
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()
        wrapper = self._load_model(approach)

        if isinstance(dm, TSDLDataModule):
            x_test, y_test = dm.get_test_windows()

        elif isinstance(dm, TSMLDataModule):
            _, _, _, _, x_test, y_test = dm.get_data()

        else:
            raise NotImplementedError(
                f"Unsupported datamodule type: {type(dm).__name__}"
            )

        preds = wrapper.predict(x_test)
        metrics = TimeSeriesEvaluator().evaluate(y_test, preds)

        print(metrics)
