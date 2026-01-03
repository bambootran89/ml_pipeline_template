"""Model training step for flexible pipeline.

Enhanced to support using best params from TuningStep.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.compat.v2.steps.base import BasePipelineStep
from mlproject.src.trainer.factory import TrainerFactory


class TrainerStep(BasePipelineStep):
    """Train a model on preprocessed data.

    This step supports:
    - Feature injection from upstream models
    - Multiple model types (ML/DL)
    - Using tuned hyperparameters from TuningStep
    - Saving trained artifacts

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Transformed features (required).
    <step_id>_features : np.ndarray, optional
        Injected features from dependency steps if output_as_feature=True.
    <tune_step_id>_best_params : Dict[str, Any], optional
        Best hyperparameters from TuningStep if use_tuned_params=True.

    Context Outputs
    ---------------
    <step_id>_model : Any
        Trained model wrapper.
    <step_id>_datamodule : DataModule
        Built datamodule.
    <step_id>_features : np.ndarray, optional
        Model predictions if output_as_feature=True.

    Configuration
    -------------
    model : str
        Model name to instantiate.
    hyperparams : dict
        Model hyperparameters.
    output_as_feature : bool, default=False
        If True, store predictions in context for downstream injection.
    use_tuned_params : bool, default=False
        If True, use best_params from upstream TuningStep.
    tune_step_id : str, optional
        ID of TuningStep (required if use_tuned_params=True).

    Examples
    --------
    # Standard training
    - id: "train_model"
      type: "model"
      enabled: true

    # Training with tuned params
    - id: "train_best"
      type: "model"
      enabled: true
      depends_on: ["tune_model"]
      use_tuned_params: true
      tune_step_id: "tune_model"
    """

    def __init__(
        self,
        *args,
        use_tuned_params: bool = False,
        tune_step_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize model training step.

        Parameters
        ----------
        use_tuned_params : bool, default=False
            Whether to use best params from TuningStep.
        tune_step_id : str, optional
            ID of TuningStep to read best_params from.
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.output_as_feature = False
        self.use_tuned_params = use_tuned_params
        self.tune_step_id = tune_step_id

    def _inject_features(
        self, df: pd.DataFrame, context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Inject features from dependency steps into dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Base preprocessed data.
        context : Dict[str, Any]
            Pipeline context containing potential feature arrays.

        Returns
        -------
        pd.DataFrame
            DataFrame with injected features added as new columns.
        """
        df_out = df.copy()

        for dep_id in self.depends_on:
            feature_key = f"{dep_id}_features"
            if feature_key in context:
                features = context[feature_key]
                if isinstance(features, np.ndarray):
                    # Add as new columns
                    for i in range(features.shape[1]):
                        df_out[f"{dep_id}_feat_{i}"] = features[:, i]
                    print(
                        f"[{self.step_id}] Injected {features.shape[1]} "
                        f"features from '{dep_id}'"
                    )

        return df_out

    def _get_hyperparams(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters (from config or tuning).

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters to use for training.
        """
        if self.use_tuned_params:
            # Use best params from TuningStep
            if self.tune_step_id is None:
                raise ValueError(
                    f"Step '{self.step_id}': use_tuned_params=True requires "
                    f"tune_step_id parameter"
                )

            best_params_key = f"{self.tune_step_id}_best_params"
            if best_params_key not in context:
                raise ValueError(
                    f"Step '{self.step_id}': Expected '{best_params_key}' in context. "
                    f"Make sure TuningStep '{self.tune_step_id}' runs before this step."
                )

            best_params = context[best_params_key]

            print(f"\n[{self.step_id}] Using tuned hyperparameters:")
            for param, value in best_params.items():
                print(f"  - {param}: {value}")

            # Merge best_params into config
            cfg_copy = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            if "args" not in cfg_copy.experiment.hyperparams:
                cfg_copy.experiment.hyperparams.args = {}

            # Update with best params
            for param, value in best_params.items():
                cfg_copy.experiment.hyperparams.args[param] = value

            return dict(cfg_copy.experiment.hyperparams)
        else:
            # Use config hyperparams
            return dict(self.cfg.experiment.hyperparams)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Train model on preprocessed data.

        Parameters
        ----------
        context : Dict[str, Any]
            Must contain 'preprocessed_data'.

        Returns
        -------
        Dict[str, Any]
            Context with trained model added.

        Raises
        ------
        RuntimeError
            If required data is missing.
        """
        self.validate_dependencies(context)

        if "preprocessed_data" not in context:
            raise RuntimeError(f"Step '{self.step_id}' requires 'preprocessed_data'")

        df: pd.DataFrame = context["preprocessed_data"]

        # Inject features from dependencies if available
        df = self._inject_features(df, context)

        # Get hyperparams (from config or tuning)
        hyperparams = self._get_hyperparams(context)

        # Build components
        model_name = self.cfg.experiment.model.lower()
        model_type = self.cfg.experiment.model_type.lower()

        # Use updated config if tuned params
        if self.use_tuned_params:
            # cfg_copy = OmegaConf.create(
            #     OmegaConf.to_container(self.cfg, resolve=True)
            # )
            if "args" not in self.cfg.experiment.hyperparams:
                self.cfg.experiment.hyperparams.args = {}
            best_params = context[f"{self.tune_step_id}_best_params"]
            for param, value in best_params.items():
                self.cfg.experiment.hyperparams.args[param] = value
            wrapper = ModelFactory.create(model_name, self.cfg)
            datamodule = DataModuleFactory.build(self.cfg, df)
        else:
            wrapper = ModelFactory.create(model_name, self.cfg)
            datamodule = DataModuleFactory.build(self.cfg, df)

        datamodule.setup()

        trainer = TrainerFactory.create(
            model_type=model_type,
            model_name=model_name,
            wrapper=wrapper,
            save_dir=self.cfg.training.artifacts_dir,
        )

        # Train
        print(f"\n[{self.step_id}] Training model...")
        trained_wrapper = trainer.train(datamodule, hyperparams)

        # Store in context
        context[f"{self.step_id}_model"] = trained_wrapper
        context[f"{self.step_id}_datamodule"] = datamodule

        # Optionally generate features for downstream steps
        if self.output_as_feature:
            if hasattr(datamodule, "get_data"):
                x_train, _, _, _, _, _ = datamodule.get_data()
            else:
                x_train, _ = datamodule.get_test_windows()

            preds = trained_wrapper.predict(x_train)
            context[f"{self.step_id}_features"] = preds

            print(f"[{self.step_id}] Generated features: {preds.shape}")

        print(f"[{self.step_id}] Model trained successfully")
        return context
