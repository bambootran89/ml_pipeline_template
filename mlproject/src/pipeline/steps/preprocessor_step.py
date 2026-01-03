"""Preprocessing step with data wiring support."""

from typing import Any, Dict, List, Optional

import pandas as pd

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.preprocess.offline import OfflinePreprocessor


class PreprocessorStep(BasePipelineStep):
    """Fit and apply preprocessing transformations.

    This step fits preprocessing on training data and transforms
    the full dataset. Supports data wiring for flexible I/O.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    df : pd.DataFrame
        Full input data (required).
    train_df : pd.DataFrame
        Training subset.
    test_df : pd.DataFrame
        Test subset.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    preprocessed_data : pd.DataFrame
        Transformed feature data (default key).
    preprocessor : OfflinePreprocessor
        Fitted preprocessor instance.

    Wiring Example
    --------------
    ::

        - id: "preprocess_v2"
          type: "preprocessor"
          wiring:
            inputs:
              df: "raw_data"           # Custom input key
            outputs:
              data: "features_v2"      # Custom output key
          is_train: true

    Configuration
    -------------
    is_train : bool, default=True
        If True, fit preprocessor on training data.
        If False, load saved preprocessor artifacts.
    """

    # Default keys for backward compatibility
    DEFAULT_INPUTS = {"df": "df", "train_df": "train_df", "test_df": "test_df"}
    DEFAULT_OUTPUTS = {"data": "preprocessed_data"}

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        is_train: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize preprocessing step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration object.
        enabled : bool, default=True
            Whether step is active.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        is_train : bool, default=True
            Whether to fit (train mode) or load (eval mode).
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
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

        Uses wiring configuration for input/output key mapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with preprocessed data and preprocessor added.

        Raises
        ------
        RuntimeError
            If required data is missing from context.
        """
        self.validate_dependencies(context)

        # Get inputs using wiring
        df: pd.DataFrame = self.get_input(context, "df")
        train_df: pd.DataFrame = (
            self.get_input(context, "train_df", required=False) or pd.DataFrame()
        )
        test_df: pd.DataFrame = (
            self.get_input(context, "test_df", required=False) or pd.DataFrame()
        )

        is_splited = context.get("is_splited_input", False)

        preprocessor = OfflinePreprocessor(is_train=self.is_train, cfg=self.cfg)

        if self.is_train:
            # TRAINING MODE: Fit on training subset
            print(f"[{self.step_id}] Training mode - fitting preprocessor")

            if "dataset" in df.columns:
                train_subset = df[df["dataset"] == "train"]
            else:
                train_subset = preprocessor.select_train_subset(df)

            preprocessor.fit_manager(train_subset)
            df_transformed = preprocessor.transform(df)

        else:
            # EVAL MODE: Load saved artifacts
            print(f"[{self.step_id}] Eval mode - loading saved preprocessor")
            preprocessor.transform_manager.load(self.cfg)

            if not is_splited:
                test_df = df.copy()

            df_transformed = preprocessor.transform(test_df)
            df_transformed = self._attach_targets_if_needed(test_df, df_transformed)

            if is_splited:
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

        # Store outputs using wiring
        self.set_output(context, "data", df_transformed, "preprocessed_data")
        context["preprocessor"] = preprocessor

        print(f"[{self.step_id}] Preprocessed data: {df_transformed.shape}")
        return context
