import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from mlproject.src.data.dl_datamodule import DLDataModule
from mlproject.src.data.ml_datamodule import MLDataModule


class TestDataModules:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Create fake DataFrame and config for all tests."""
        n = 50
        # fake features theo thứ tự thời gian
        idx = pd.date_range("2020-01-01", periods=200, freq="H")
        self.df = pd.DataFrame(
            {
                "HUFL": np.sin(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "MUFL": np.cos(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "mobility_inflow": np.random.rand(len(idx)) * 10,
            },
            index=idx,
        )
        self.df.index.name = "date"

        self.cfg = {
            "preprocessing": {"split": {"train": 0.6, "val": 0.2, "test": 0.2}},
            "training": {"num_workers": 0},
        }
        self.target_column = ["HUFL", "MUFL", "mobility_inflow"]

    def test_dl_datamodule(self):
        """Test DLDataModule creates DataLoaders and windows correctly."""
        dl_module = DLDataModule(self.df, self.cfg, self.target_column)
        dl_module.setup(input_chunk=5, output_chunk=2, batch_size=8, num_workers=0)
        train_loader, val_loader, input_chunk, output_chunk = dl_module.get_loaders()

        # Check types
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert input_chunk == 5
        assert output_chunk == 2

        # Check one batch
        for x_batch, y_batch in train_loader:
            # x_batch shape: (batch_size, input_chunk, n_features)
            assert x_batch.shape[1] == input_chunk
            assert x_batch.shape[2] == 3  # number of features
            # y_batch shape: (batch_size, output_chunk)
            assert y_batch.shape[1] == output_chunk
            break

        # Test test windows
        x_test, y_test = dl_module.get_test_windows()
        assert x_test.shape[1] == input_chunk
        assert x_test.shape[2] == 3
        assert y_test.shape[1] == output_chunk

    def test_ml_datamodule(self):
        """Test MLDataModule returns correct numpy arrays for train/val/test."""
        # chọn HUFL làm target cho ML

        ml_module = MLDataModule(self.df, self.cfg, target_column=self.target_column)
        ml_module.setup(
            input_chunk=5,
            output_chunk=2,
        )
        # setup với window length để có consistency với DLDataModule
        X_train, y_train, X_val, y_val, X_test, y_test = ml_module.get_data()

        # Test test windows
        assert X_test.shape[1] == 5
        assert X_test.shape[2] == 3
        assert y_test.shape[1] == 2
