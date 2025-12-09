from typing import Any, Dict, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.datamodule.tsml import TSMLDataModule


class DataModuleFactory:
    """
    Factory that constructs the correct DataModule based on model type.
    """

    @staticmethod
    def build(cfg: DictConfig, df: pd.DataFrame):
        """
        Build an appropriate DataModule (TSDLDataModule or TSMLDataModule)
        depending on the model type specified in the experiment config.

        Args:
            cfg (DictConfig): Hydra/OmegaConf configuration object.
            df (pd.DataFrame): Input dataframe after preprocessing.

        Returns:
            TSDLDataModule | TSMLDataModule:
                - TSDLDataModule for deep learning models
                - TSMLDataModule for classical ML models

        Raises:
            ValueError: If the model type is unknown or unsupported.
        """
        model_type = cfg.experiment.get("model", "").lower()
        target_col = cfg.data.get("target_columns", [""])[0]

        input_chunk = cfg.experiment.hyperparams.get("input_chunk_length", -1)
        output_chunk = cfg.experiment.hyperparams.get("output_chunk_length", -1)

        dl_models = {"tft", "nlinear", "lstm", "gru", "transformer"}
        ml_models = {"xgboost", "xgb", "lgbm", "lightgbm", "rf", "svm", "sklearn"}

        # ---- DL ---- #
        if model_type in dl_models:
            return TSDLDataModule(
                df=df,
                cfg=cfg,
                target_column=target_col,
                input_chunk=input_chunk,
                output_chunk=output_chunk,
            )

        # ---- ML ---- #
        if model_type in ml_models:
            raw_cfg = OmegaConf.to_container(cfg, resolve=True)

            # mypy guarantee
            assert isinstance(raw_cfg, dict)

            # Cast to Dict[str, Any]
            cfg_dict = cast(Dict[str, Any], raw_cfg)

            return TSMLDataModule(
                df=df,
                cfg=cfg_dict,
                target_column=target_col,
                input_chunk=input_chunk,
                output_chunk=output_chunk,
            )

        raise ValueError(
            f"[DataModuleFactory] Unknown model type: {model_type}. "
            f"Supported DL={dl_models}, ML={ml_models}"
        )
