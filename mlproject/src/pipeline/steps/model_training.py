"""Model training step for flexible pipeline."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.trainer.factory import TrainerFactory


class ModelTrainingStep(BasePipelineStep):
    """Train a model on preprocessed data.

    This step supports:
    - Feature injection from upstream models
    - Multiple model types (ML/DL)
    - Saving trained artifacts

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Transformed features (required).
    <step_id>_features : np.ndarray, optional
        Injected features from dependency steps if output_as_feature=True.

    Context Outputs
    ---------------
    <step_id>_model : Any
        Trained model wrapper.
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
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize model training step."""
        super().__init__(*args, **kwargs)
        self.output_as_feature = False

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

        # Build components
        model_name = self.cfg.experiment.model.lower()
        model_type = self.cfg.experiment.model_type.lower()

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
        hyperparams = dict(self.cfg.experiment.hyperparams)
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
