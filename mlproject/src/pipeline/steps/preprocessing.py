"""Preprocessing step for flexible pipeline."""

from typing import Any, Dict

import pandas as pd

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.preprocess.offline import OfflinePreprocessor


class PreprocessingStep(BasePipelineStep):
    """Fit and apply preprocessing transformations.

    This step fits preprocessing on training data and transforms
    the full dataset.

    Context Inputs
    --------------
    raw_data : pd.DataFrame
        Raw input data (required).

    Context Outputs
    ---------------
    preprocessed_data : pd.DataFrame
        Transformed feature data.
    preprocessor : OfflinePreprocessor
        Fitted preprocessor instance.
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fit preprocessor and transform data.

        Parameters
        ----------
        context : Dict[str, Any]
            Must contain 'raw_data' key.

        Returns
        -------
        Dict[str, Any]
            Context with 'preprocessed_data' and 'preprocessor' added.

        Raises
        ------
        RuntimeError
            If 'raw_data' is missing from context.
        """
        self.validate_dependencies(context)

        if "raw_data" not in context:
            raise RuntimeError(f"Step '{self.step_id}' requires 'raw_data' in context")

        df: pd.DataFrame = context["raw_data"]

        preprocessor = OfflinePreprocessor(is_train=True, cfg=self.cfg)

        # Fit on training subset
        if "dataset" in df.columns:
            train_df = df[df["dataset"] == "train"]
        else:
            train_df = preprocessor.select_train_subset(df)

        preprocessor.fit_manager(train_df)

        # Transform full dataset
        df_transformed = preprocessor.transform(df)

        context["preprocessed_data"] = df_transformed
        context["preprocessor"] = preprocessor

        print(f"[{self.step_id}] Preprocessed data: {df_transformed.shape}")
        return context
