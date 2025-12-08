import pandas as pd
from omegaconf import DictConfig

from mlproject.src.datamodule.tsdl import TSDLDataModule


class DataModuleFactory:
    """
    Factory class that builds a fully configured TSDLDataModule.

    Ensures consistent initialization across:
        - TrainingPipeline
        - EvalPipeline
        - TestPipeline

    This avoids duplicated logic and ensures that all pipelines
    use the same windowing configuration.
    """

    @staticmethod
    def build(cfg: DictConfig, df: pd.DataFrame) -> TSDLDataModule:
        """
        Construct a TSDLDataModule with correct configuration parameters.

        Args:
            cfg (DictConfig): Global config.
            df (pd.DataFrame): Preprocessed dataset.

        Returns:
            TSDLDataModule: Not yet setup() — caller decides modes.
        """
        target_col = cfg.data.get("target_columns", ["HUFL"])[0]

        return TSDLDataModule(
            df=df,
            cfg=cfg,
            target_column=target_col,
            input_chunk=cfg.training.get("input_chunk", 24),
            output_chunk=cfg.training.get("output_chunk", 6),
        )
