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

    Responsibilities:
    - Load preprocessed dataset (or transform using SAVED scaler)
    - Load trained model weights
    - Evaluate model

    IMPORTANT: Does NOT fit scaler, only transforms using saved scaler.
    """

    def __init__(self, cfg_path=""):
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

    def preprocess(self):
        """
        Transform data using SAVED scaler (does NOT fit).

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        preprocessor = OfflinePreprocessor(self.cfg)

        # Load raw data
        df = preprocessor.load_raw_data()

        # Transform only (using saved scaler)
        df = preprocessor.transform(df)

        return df

    def _load_model(self, approach):
        """
        Load trained model wrapper from artifacts.

        Args:
            approach: Experiment approach config.

        Returns:
            Loaded model wrapper.
        """
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
        Evaluate a single approach.

        Args:
            approach: Experiment approach config.
            data: Preprocessed dataset (transformed using saved scaler).
        """
        df = data

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
