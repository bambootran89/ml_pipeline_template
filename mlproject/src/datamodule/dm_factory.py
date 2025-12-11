from typing import Any, Dict, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# from mlproject.src.datamodule.tsbase import TSBaseDataModule
# from mlproject.src.datamodule.tsdl import TSDLDataModule
from mlproject.src.utils.factory_base import DynamicFactoryBase

class DataModuleFactory(DynamicFactoryBase):
    """
    Factory that constructs the correct DataModule based on model type.
    """

    @staticmethod
    def build(cls, cfg: DictConfig, df: pd.DataFrame):
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
        if model_type in dl_models:
            # DL DataModule
            config_entry = {
                "module": "mlproject.src.datamodule.tsdl", 
                "class": "TSDLDataModule"
            }
            cfg_init = cfg 
        else:
            # ML DataModule (TSBaseDataModule)
            config_entry = {
                "module": "mlproject.src.datamodule.tsbase", 
                "class": "TSBaseDataModule"
            }
            raw_cfg = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(raw_cfg, dict)
            cfg_init = cast(Dict[str, Any], raw_cfg)


        DataModuleClass = cls._get_class_from_config(config_entry)
        return DataModuleClass(
            df=df,
            cfg=cfg_init,
            target_column=target_col,
            input_chunk=input_chunk,
            output_chunk=output_chunk,
        )

        
        # # ---- DL ---- #
        # if model_type in dl_models:
        #     return TSDLDataModule(
        #         df=df,
        #         cfg=cfg,
        #         target_column=target_col,
        #         input_chunk=input_chunk,
        #         output_chunk=output_chunk,
        #     )
        # raw_cfg = OmegaConf.to_container(cfg, resolve=True)
        # # mypy guarantee
        # assert isinstance(raw_cfg, dict)

        # # Cast to Dict[str, Any]
        # cfg_dict = cast(Dict[str, Any], raw_cfg)
        # return TSBaseDataModule(
        #     df=df,
        #     cfg=cfg_dict,
        #     target_column=target_col,
        #     input_chunk=input_chunk,
        #     output_chunk=output_chunk,
        # )

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        """Alias  """

        return cls.build(*args, **kwargs)
