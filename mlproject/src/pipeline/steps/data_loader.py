"""Data loading step for flexible pipeline."""

from typing import Any, Dict

import pandas as pd

from mlproject.src.datamodule.loader import resolve_datasets_from_cfg
from mlproject.src.pipeline.steps.base import BasePipelineStep


class DataLoaderStep(BasePipelineStep):
    """Load raw data from configured source.

    This step loads data using the existing datamodule loader system
    and stores the result in context under 'raw_data'.

    Context Outputs
    ---------------
    raw_data : pd.DataFrame
        Loaded raw dataset.
    """

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from configuration.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context (unused for this step).

        Returns
        -------
        Dict[str, Any]
            Context with 'raw_data' key added.
        """
        df, train_df, val_df, test_df = resolve_datasets_from_cfg(self.cfg)

        # Consolidate into single DataFrame if needed
        if len(df) == 0 and len(train_df) > 0:
            train_df["dataset"] = "train"
            val_df["dataset"] = "val"
            test_df["dataset"] = "test"
            df = pd.concat([train_df, val_df, test_df], axis=0)

        context["raw_data"] = df
        print(f"[{self.step_id}] Loaded data: {df.shape}")
        return context
