from mlproject.src.datamodule.dm_factory import DataModuleFactory
from mlproject.src.eval.evaluator import mae, mse, smape
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.offline import OfflinePreprocessor


class EvalPipeline(BasePipeline):
    """
    Evaluation-only pipeline.

    Loads:
        - Preprocessed dataset (or recompute)
        - DataModule
        - Saved model weights
    """

    def __init__(self, cfg_path=""):
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self):
        """Run preprocessing and return dataset for evaluation."""
        return OfflinePreprocessor(self.cfg).run()

    def _load_model(self, approach):
        """Load the trained model wrapper from artifacts."""
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

    def run_approach(self, approach, data):
        """
        Run evaluation for a single approach.

        Args:
            approach (dict): Experiment approach config.
            data (pd.DataFrame): Preprocessed dataset.
        """
        df = data  # rename locally for clarity

        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()

        x_test, y_test = dm.get_test_windows()

        wrapper = self._load_model(approach)
        preds = wrapper.predict(x_test)

        print(
            f"[EVAL] MAE={mae(y_test, preds):.6f}, "
            f"MSE={mse(y_test, preds):.6f}, "
            f"SMAPE={smape(y_test, preds):.6f}"
        )
