import os
from typing import Dict, List

import numpy as np
import pandas as pd

from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import test_preprocess_request

ARTIFACTS_DIR = os.path.join("mlproject", "artifacts", "models")
CONFIG_PATH = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")


class ModelService:
    """
    Service managing the model and configuration for time-series forecasting inference.

    Responsibilities:
        1. Load configuration from YAML file.
        2. Load a trained model (NLinearWrapper or TFTWrapper) from artifacts.
        3. Preprocess raw input data into model-ready features.
        4. Prepare the input window of appropriate length.
        5. Perform predictions and return flattened output.

    Attributes:
        model: The loaded model wrapper instance.
        cfg: The OmegaConf configuration object.
        input_chunk_length: Number of timesteps required for the input window.
    """

    def __init__(self):
        """
        Initialize an empty ModelService instance.

        The model, configuration, and input_chunk_length are None until
        `load_config` and `load_model` are called.
        """
        self.model = None
        self.cfg = None
        self.input_chunk_length = None

    def load_config(self, cfg_path: str = CONFIG_PATH):
        """
        Load configuration from a YAML file and set input window length.

        Args:
            cfg_path (str): Path to the experiment config YAML file.

        Side Effects:
            Sets `self.cfg` and `self.input_chunk_length`.
        """
        self.cfg = ConfigLoader.load(cfg_path)
        experiment = self.cfg.experiment
        self.input_chunk_length = experiment.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )

    def load_model(self):
        """
        Load the trained model from the artifacts directory.

        Raises:
            RuntimeError: If configuration has not been loaded.
            RuntimeError: If the model type is unknown.

        Side Effects:
            Sets `self.model` to the loaded model wrapper.
        """
        if self.cfg is None:
            raise RuntimeError("Config not loaded")

        experiment = self.cfg.experiment
        model_name = experiment.get("model")
        hp = experiment.get("hyperparams", {})

        if model_name == "nlinear":
            wrapper = NLinearWrapper(hp)
        elif model_name == "tft":
            wrapper = TFTWrapper(hp)
        else:
            raise RuntimeError(f"Unknown model {model_name}")

        wrapper.load(ARTIFACTS_DIR)
        self.model = wrapper

    def preprocess(self, data_dict: Dict[str, List]) -> pd.DataFrame:
        """
        Convert raw input dictionary to a preprocessed DataFrame for inference.

        Args:
            data_dict (Dict[str, List]): Dictionary containing raw historical
                                         data. Keys are feature names, values
                                         are lists of feature values.

        Returns:
            pd.DataFrame: Transformed features ready for model input.

        Side Effects:
            Uses the loaded configuration (`self.cfg`) and preprocessing
            utilities from `test_preprocess_request`.
        """
        df = pd.DataFrame(data_dict)
        if "date" in df.columns:
            df = df.set_index("date")
        return test_preprocess_request(df, self.cfg)

    def prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the last `input_chunk_length` rows from transformed data.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame.

        Returns:
            np.ndarray: 3D array shaped [1, seq_len, n_features] ready for
                        model prediction.

        Raises:
            ValueError: If `df` has fewer rows than `self.input_chunk_length`.
        """
        if len(df) < self.input_chunk_length:
            raise ValueError(
                f"Input data has {len(df)} rows,\
                      need at least {self.input_chunk_length}"
            )
        return df.iloc[-self.input_chunk_length :].values[np.newaxis, :]

    def predict(self, data_dict: Dict[str, List]) -> list:
        """
        Perform inference on raw input data.

        Args:
            data_dict (Dict[str, List]): Raw input dictionary containing
                                         historical features.

        Returns:
            list: Flattened list of predicted values.

        Raises:
            Any exception raised by preprocessing, input window preparation,
            or the model's predict method.
        """
        df_transformed = self.preprocess(data_dict)
        x_input = self.prepare_input_window(df_transformed)
        preds = self.model.predict(x_input)
        return preds.flatten().tolist()
