"""Flexible training pipeline using configuration-driven execution."""

from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.executor import PipelineExecutor
from mlproject.src.utils.config_loader import ConfigLoader


class FlexibleTrainingPipeline(BasePipeline):
    """Configuration-driven flexible training pipeline.

    This pipeline replaces hardcoded logic with YAML-configured steps
    that can be composed, reordered, and conditionally enabled.

    Example YAML Configuration
    ---------------------------
    pipeline:
      name: "two_stage_kmeans_xgb"
      steps:
        - id: load_data
          type: data_loader
          enabled: true

        - id: preprocess
          type: preprocessor
          enabled: true
          depends_on: [load_data]

        - id: cluster
          type: model
          enabled: true
          depends_on: [preprocess]
          output_as_feature: true

        - id: train
          type: model
          enabled: true
          depends_on: [preprocess, cluster]

        - id: evaluate
          type: evaluator
          enabled: true
          depends_on: [train]
          model_step_id: train

        - id: log
          type: logger
          enabled: true
          depends_on: [train, evaluate]
          model_step_id: train
          eval_step_id: evaluate
    """

    def __init__(self, cfg_path: str = "") -> None:
        """Initialize flexible pipeline.

        Parameters
        ----------
        cfg_path : str
            Path to configuration file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.executor = PipelineExecutor(self.cfg)

    def preprocess(self) -> Optional[pd.DataFrame]:
        """Not used in flexible pipeline.

        Returns
        -------
        None
            Preprocessing is handled by pipeline steps.
        """
        return None

    def run_exp(self, data: Any = None) -> Dict[str, Any]:
        """Execute configured pipeline.

        Parameters
        ----------
        data : Any, optional
            Unused. Data loading is handled by pipeline steps.

        Returns
        -------
        Dict[str, Any]
            Final pipeline context containing all outputs.
        """
        context = self.executor.execute()
        return context
