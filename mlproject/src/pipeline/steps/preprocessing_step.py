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

    Configuration Parameters
    ------------------------
    is_train : bool, default=True
        If True, fit preprocessor on training data.
        If False, load saved preprocessor artifacts.
    """

    def __init__(self, *args, is_train: bool = True, **kwargs) -> None:
        """Initialize preprocessing step.

        Parameters
        ----------
        is_train : bool, default=True
            Whether to fit (train mode) or load (eval mode).
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.is_train = is_train

    def _attach_targets_if_needed(
        self, df_raw: pd.DataFrame, fea_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attach target columns back for tabular datasets.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data.
        fea_df : pd.DataFrame
            Transformed features.

        Returns
        -------
        pd.DataFrame
            Final dataset for evaluation.
        """
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", "timeseries")).lower()

        if data_type == "timeseries":
            return fea_df

        target_cols = data_cfg.get("target_columns", [])
        tar_df = df_raw[target_cols]

        return pd.concat([fea_df, tar_df], axis=1)

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

        df: pd.DataFrame = context["df"]
        train_df: pd.DataFrame = context["train_df"]
        # val_df: pd.DataFrame = context["val_df"]
        test_df: pd.DataFrame = context["test_df"]

        preprocessor = OfflinePreprocessor(is_train=self.is_train, cfg=self.cfg)

        if self.is_train:
            # TRAINING MODE: Fit on training subset
            print(f"[{self.step_id}] Training mode - fitting preprocessor")

            if "dataset" in df.columns:
                train_df = df[df["dataset"] == "train"]
            else:
                train_df = preprocessor.select_train_subset(df)

            preprocessor.fit_manager(train_df)
            df_transformed = preprocessor.transform(df)

        else:
            # EVAL MODE: Load saved artifacts
            print(f"[{self.step_id}] Eval mode - loading saved preprocessor")
            preprocessor.transform_manager.load(self.cfg)

            if not context["is_splited_input"]:
                test_df = df.copy()

            df_transformed = preprocessor.transform(test_df)

            df_transformed = self._attach_targets_if_needed(test_df, df_transformed)
            if context["is_splited_input"]:
                df_transformed["dataset"] = "test"
                print(
                    "[DataCheck] Test split already performed upstream "
                    "→ assigning dataset='test' label."
                )
            else:
                print(
                    "[DataCheck] Dataset column missing "
                    "→ test data is generated via sliding windows "
                    "+ ratio split from config."
                )

        context["preprocessed_data"] = df_transformed
        context["preprocessor"] = preprocessor

        print(f"[{self.step_id}] Preprocessed data: {df_transformed.shape}")
        return context
