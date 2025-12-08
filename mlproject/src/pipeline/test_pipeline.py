from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import test_preprocess_request


class TestPipeline(BasePipeline):
    """
    Inference / online serving pipeline.
    """

    def __init__(self, cfg_path=""):
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self, data=None):
        """
        Lightweight deterministic preprocessing for real-time inference.
        `data` is optional for compatibility with BasePipeline.
        """
        if data is None:
            raise ValueError("Preprocess requires input data for inference")
        return test_preprocess_request(data, self.cfg)

    def _load_model(self, approach):
        name = approach["model"]
        hp = approach.get("hyperparams", {})

        if name == "nlinear":
            wrapper = NLinearWrapper(hp)
        elif name == "tft":
            wrapper = TFTWrapper(hp)
        else:
            raise RuntimeError(f"Unknown model {name}")

        wrapper.load(self.cfg.training.artifacts_dir)
        return wrapper

    def run_approach(self, approach, data):  # pylint: disable=arguments-renamed
        """Run inference on one approach."""
        raw_df = data
        df = self.preprocess(raw_df)

        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()  # inference window handled internally
        x_test, _ = dm.get_test_windows()
        x_latest = x_test[-1:]  # take last window

        wrapper = self._load_model(approach)
        preds = wrapper.predict(x_latest)
        print(preds)
        return preds
