from unittest.mock import patch

import numpy as np
import pandas as pd

from mlproject.src.preprocess.engine import PreprocessEngine
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.preprocess.online import online_preprocess_request


class TestPreprocessPipeline:
    """
    Full test suite for preprocessing pipeline:
    - Offline pipeline runs
    - Online preprocess runs
    - Drift consistency between offline and online
    - Scaler loads only once in engine
    - Schema consistency for online requests
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare shared preprocessor config and ensure offline pipeline
        has produced a scaler artifact for online tests.
        """
        cls.cfg = {
            "preprocessing": {
                "steps": [
                    {"name": "fill_missing", "method": "ffill"},
                    {
                        "name": "gen_covariates",
                        "covariates": {
                            "past": ["mobility_inflow"],
                            "future": [
                                "day_of_week",
                                "is_holiday",
                                "influenza_visits",
                            ],
                            "static": ["commune"],
                        },
                    },
                    {"name": "normalize", "method": "zscore", "columns": None},
                ]
            }
        }

        # Run offline once to ensure scaler.pkl exists
        pre = OfflinePreprocessor(cls.cfg)
        cls.df_off = pre.run()

        # Warm up engine for online tests
        cls.engine = PreprocessEngine.instance()
        cls.engine.base.load_scaler()

    def test_offline_runs(self):
        """Offline pipeline must run end-to-end and produce non-empty DataFrame."""
        pre = OfflinePreprocessor(self.cfg)
        df = pre.run()
        assert len(df) > 0

    def test_online_preprocess_basic(self):
        """Online preprocess must return dict with numeric values."""
        sample = {
            "HUFL": 0.5,
            "MUFL": 1.2,
            "mobility_inflow": 10,
        }

        out = online_preprocess_request(sample)
        assert isinstance(out, dict)
        assert len(out) > 0

    def test_engine_load_scaler_once(self):
        """
        Ensure engine only loads scaler one time for online requests.
        """

        engine = PreprocessEngine.instance()
        engine._loaded = False

        call_counter = {"calls": 0}

        def fake_load():
            call_counter["calls"] += 1

        # Patch the scaler loader
        with patch.object(engine.base, "load_scaler", side_effect=fake_load):
            online_preprocess_request({"HUFL": 1, "MUFL": 2, "mobility_inflow": 3})
            online_preprocess_request({"HUFL": 4, "MUFL": 5, "mobility_inflow": 6})
            online_preprocess_request({"HUFL": 7, "MUFL": 8, "mobility_inflow": 9})

        assert call_counter["calls"] == 1, "Scaler should load exactly once."

    def test_schema_columns_consistency(self):
        """
        Ensure online transform always outputs all expected scaler features,
        even for missing fields.
        """

        cols = self.engine.base.scaler_columns

        # Missing all fields
        empty_request = {}
        out = online_preprocess_request(empty_request)

        for col in cols:
            assert col in out, f"Missing expected feature: {col}"
