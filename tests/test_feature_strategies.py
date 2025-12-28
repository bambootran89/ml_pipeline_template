"""
Unit tests for feature retrieval strategies.

Tests cover strategy creation, timeseries retrieval,
tabular retrieval, and facade integration.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.features.strategies import (
    StrategyFactory,
    TabularRetrievalStrategy,
    TimeseriesRetrievalStrategy,
)


class TestStrategyFactory:
    """Test suite for StrategyFactory."""

    def test_create_timeseries_strategy(self):
        """Factory creates timeseries strategy correctly."""
        strategy = StrategyFactory.create("timeseries")
        assert isinstance(strategy, TimeseriesRetrievalStrategy)

    def test_create_tabular_strategy(self):
        """Factory creates tabular strategy correctly."""
        strategy = StrategyFactory.create("tabular")
        assert isinstance(strategy, TabularRetrievalStrategy)

    def test_create_invalid_type_raises(self):
        """Factory raises on unknown data type."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            StrategyFactory.create("invalid_type")


class TestTimeseriesStrategy:
    """Test suite for timeseries retrieval strategy."""

    @patch("mlproject.src.features.strategies.TimeSeriesFeatureStore")
    def test_retrieve_calls_store_correctly(self, mock_ts_store_class):
        """Strategy delegates to TimeSeriesFeatureStore properly."""
        mock_store = Mock()
        mock_ts_store = Mock()
        mock_ts_store_class.return_value = mock_ts_store

        expected_df = pd.DataFrame({"value": [1, 2, 3]})
        mock_ts_store.get_sequence_by_range.return_value = expected_df

        strategy = TimeseriesRetrievalStrategy()
        config = {
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2024-01-02T00:00:00+00:00",
        }

        result = strategy.retrieve(
            store=mock_store,
            features=["view:feat1"],
            entity_key="location_id",
            entity_id=1,
            config=config,
        )

        assert result.equals(expected_df)
        mock_ts_store.get_sequence_by_range.assert_called_once()


class TestFeatureStoreFacade:
    """Test suite for FeatureStoreFacade."""

    def test_facade_validates_missing_config(self):
        """Facade raises when required config keys are missing."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://repo",
                    "type": "timeseries",
                    # Missing: featureview, features
                }
            }
        )

        facade = FeatureStoreFacade(cfg)

        with pytest.raises(ValueError, match="Missing config keys"):
            facade.load_features()

    @patch("mlproject.src.features.facade.FeatureStoreFactory")
    @patch("mlproject.src.features.facade.StrategyFactory")
    def test_facade_loads_timeseries(self, mock_strategy_factory, mock_store_factory):
        """Facade successfully loads timeseries features."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://test_repo",
                    "type": "timeseries",
                    "featureview": "test_view",
                    "features": ["feat1", "feat2"],
                    "entity_key": "id",
                    "entity_id": 1,
                    "start_date": "2024-01-01T00:00:00+00:00",
                    "end_date": "2024-01-02T00:00:00+00:00",
                    "index_col": "timestamp",
                }
            }
        )

        mock_store = Mock()
        mock_store_factory.create.return_value = mock_store

        mock_strategy = Mock()
        expected_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
                "feat1": [1, 2, 3],
                "feat2": [4, 5, 6],
            }
        )
        mock_strategy.retrieve.return_value = expected_df
        mock_strategy_factory.create.return_value = mock_strategy

        facade = FeatureStoreFacade(cfg)
        result = facade.load_features()

        assert "timestamp" in result.index.name or len(result) > 0
        mock_store_factory.create.assert_called_once_with(
            store_type="feast",
            repo_path="test_repo",
        )
        mock_strategy_factory.create.assert_called_once_with("timeseries")
