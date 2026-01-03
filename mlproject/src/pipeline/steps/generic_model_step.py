"""
Generic model step with flexible data wiring.

This module provides a unified model step that supports any model type
with configurable input/output routing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.context_router import create_router_from_config
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.trainer.factory import TrainerFactory


class GenericModelStep(BasePipelineStep):
    """
    Generic model step supporting any registered model with data wiring.

    This step provides a unified interface for training, prediction,
    and feature generation from any ML/DL model.

    Key Features:
    - Configurable input/output routing
    - Support for output_as_feature mode
    - Model-agnostic implementation
    - Hyperparameter override from step config

    Context Inputs
    --------------
    data : pd.DataFrame
        Training/prediction data (key configurable via wiring).

    Context Outputs
    ---------------
    model : Any
        Trained model wrapper.
    datamodule : Any
        Built datamodule.
    predictions : np.ndarray, optional
        Model predictions (if output_as_feature=True).
    features : np.ndarray, optional
        Feature array for downstream steps.

    Examples
    --------
    Clustering for feature engineering::

        - id: "kmeans_features"
          type: "generic_model"
          enabled: true
          depends_on: ["preprocess"]
          model_name: "kmean"
          output_as_feature: true
          wiring:
            inputs:
              data: "preprocessed_data"
            outputs:
              features: "cluster_labels"
              model: "kmeans_model"

    Training with custom hyperparams::

        - id: "xgb_train"
          type: "generic_model"
          model_name: "xgboost"
          hyperparams:
            n_estimators: 200
            max_depth: 8
          wiring:
            inputs:
              data: "feature_data"
            outputs:
              model: "final_model"
              predictions: "xgb_preds"
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        output_as_feature: bool = False,
        hyperparams: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize generic model step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Full experiment configuration.
        enabled : bool, default=True
            Whether step should execute.
        depends_on : Optional[List[str]], default=None
            IDs of prerequisite steps.
        model_name : Optional[str], default=None
            Model name from registry. Falls back to cfg.experiment.model.
        model_type : Optional[str], default=None
            Model type (ml/dl). Falls back to cfg.experiment.model_type.
        output_as_feature : bool, default=False
            If True, store predictions as features for downstream steps.
        hyperparams : Optional[Dict[str, Any]], default=None
            Override hyperparameters for this step.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)

        self.model_name = model_name or cfg.experiment.get("model", "").lower()
        self.model_type = model_type or cfg.experiment.get("model_type", "ml").lower()
        self.output_as_feature = output_as_feature
        self.override_hyperparams = hyperparams or {}

        # Build router from kwargs (contains wiring config)
        self.router = create_router_from_config(step_id, kwargs)

    def _get_effective_hyperparams(self) -> Dict[str, Any]:
        """
        Merge base hyperparams with step-level overrides.

        Returns
        -------
        Dict[str, Any]
            Effective hyperparameters.
        """
        base = dict(self.cfg.experiment.get("hyperparams", {}))
        base.update(self.override_hyperparams)
        return base

    def _get_input_data(self, context: Dict[str, Any]) -> pd.DataFrame:
        """
        Retrieve input data from context using router.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        pd.DataFrame
            Input data for model.
        """
        # Try configured input first
        data = self.router.get_input(
            context, "data", default_key="preprocessed_data", required=False
        )

        if data is not None:
            return data

        # Fallback to common keys
        fallback_keys = ["preprocessed_data", "df", "train_df"]
        for key in fallback_keys:
            if key in context:
                return context[key]

        raise KeyError(
            f"Step '{self.step_id}' could not find input data. "
            f"Available keys: {list(context.keys())}"
        )

    def _inject_upstream_features(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Inject feature arrays from upstream steps into dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Base dataframe.
        context : Dict[str, Any]
            Pipeline context with potential feature arrays.

        Returns
        -------
        pd.DataFrame
            DataFrame with injected features.
        """
        df_out = df.copy()

        # Check for explicitly wired feature inputs
        features_input = self.router.get_input(context, "features", required=False)

        if features_input is not None and isinstance(features_input, np.ndarray):
            # Add as columns
            if features_input.ndim == 1:
                df_out[f"{self.step_id}_input_feat"] = features_input
            else:
                for i in range(features_input.shape[1]):
                    df_out[f"{self.step_id}_input_feat_{i}"] = features_input[:, i]

            print(f"[{self.step_id}] Injected {features_input.shape} features")

        # Also check depends_on for feature outputs
        for dep_id in self.depends_on:
            feature_key = f"{dep_id}_features"
            if feature_key in context:
                features = context[feature_key]
                if isinstance(features, np.ndarray):
                    if features.ndim == 1:
                        df_out[f"{dep_id}_feat"] = features
                    else:
                        for i in range(features.shape[1]):
                            df_out[f"{dep_id}_feat_{i}"] = features[:, i]

        return df_out

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model training/prediction.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Updated context with model outputs.
        """
        self.validate_dependencies(context)

        print(f"[{self.step_id}] Executing model: {self.model_name}")

        # Get and prepare data
        df = self._get_input_data(context)
        df = self._inject_upstream_features(df, context)

        # Get hyperparams
        hyperparams = self._get_effective_hyperparams()

        # Build model with possibly updated config
        step_cfg = self._build_step_config(hyperparams)

        wrapper = ModelFactory.create(self.model_name, step_cfg)

        # Build datamodule
        datamodule = DataModuleFactory.build(step_cfg, df)
        datamodule.setup()

        # Build trainer
        trainer = TrainerFactory.create(
            model_type=self.model_type,
            model_name=self.model_name,
            wrapper=wrapper,
            save_dir=self.cfg.training.get("artifacts_dir", "artifacts/models"),
        )

        # Train
        print(f"[{self.step_id}] Training with hyperparams: {hyperparams}")
        trained_wrapper = trainer.train(datamodule, hyperparams)

        # Store outputs using router
        self.router.set_output(context, "model", trained_wrapper)
        self.router.set_output(context, "datamodule", datamodule)

        # Generate features if requested
        if self.output_as_feature:
            features = self._generate_features(trained_wrapper, datamodule)
            self.router.set_output(context, "features", features)
            print(f"[{self.step_id}] Generated features: {features.shape}")

        print(f"[{self.step_id}] Model step completed")
        return context

    def _build_step_config(
        self,
        hyperparams: Dict[str, Any],
    ) -> DictConfig:
        """
        Build config for this step with hyperparams.

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Effective hyperparameters.

        Returns
        -------
        DictConfig
            Step-specific configuration.
        """
        # Create a copy of base config
        step_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))

        # Override experiment model
        step_cfg.experiment.model = self.model_name
        step_cfg.experiment.model_type = self.model_type

        # Update hyperparams
        if "hyperparams" not in step_cfg.experiment:
            step_cfg.experiment.hyperparams = {}
        step_cfg.experiment.hyperparams.update(hyperparams)

        return step_cfg

    def _generate_features(
        self,
        wrapper: Any,
        datamodule: Any,
    ) -> np.ndarray:
        """
        Generate features from trained model.

        Parameters
        ----------
        wrapper : Any
            Trained model wrapper.
        datamodule : Any
            Data module with training data.

        Returns
        -------
        np.ndarray
            Feature array.
        """
        # Get training data for feature generation
        if hasattr(datamodule, "get_data"):
            x_train, _, _, _, _, _ = datamodule.get_data()
        elif hasattr(datamodule, "get_test_windows"):
            x_train, _ = datamodule.get_test_windows()
        else:
            raise AttributeError("DataModule does not support data extraction")

        # Generate predictions as features
        predictions = wrapper.predict(x_train)

        # Ensure 2D output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions


class ClusteringModelStep(GenericModelStep):
    """
    Specialized step for clustering models.

    Extends GenericModelStep with clustering-specific defaults
    and automatic feature generation.
    """

    def __init__(
        self,
        step_id: str,
        cfg: DictConfig,
        **kwargs: Any,
    ) -> None:
        """
        Initialize clustering step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Full experiment configuration.
        **kwargs
            Passed to GenericModelStep.
        """
        # Default to output_as_feature=True for clustering
        kwargs.setdefault("output_as_feature", True)
        kwargs.setdefault("model_name", "kmean")
        kwargs.setdefault("model_type", "ml")

        super().__init__(step_id, cfg, **kwargs)
