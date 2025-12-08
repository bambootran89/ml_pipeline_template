from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from mlproject.src.datamodele.dataset import NumpyWindowDataset
from mlproject.src.datamodele.tsbase import TSBaseDataModule


class TSDLDataModule(TSBaseDataModule):
    """
    DL DataModule for PyTorch models.
    Creates windowed dataset + DataLoader using dataloader.py utilities.
    """

    def __init__(
        self,
        df,
        cfg,
        target_column: str,
        input_chunk: int,
        output_chunk: int,
    ):
        """
        Initialize DLDataModule.

        Args:
            df (pd.DataFrame): Full dataset
            cfg (dict): Configuration dict
            target_column (str): Name of target column
        """
        super().__init__(df, cfg, target_column, input_chunk, output_chunk)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def setup(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
    ):
        """
        Prepare train/val DataLoaders from windowed datasets.

        Args:
            input_chunk (int): Length of input sequence
            output_chunk (int): Length of output sequence
            batch_size (int): Batch size for DataLoader
            num_workers (int): Number of workers for DataLoader
        """
        assert isinstance(self.x_train, np.ndarray)
        assert isinstance(self.y_train, np.ndarray)

        self.train_loader = DataLoader(
            NumpyWindowDataset(self.x_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        assert isinstance(self.x_val, np.ndarray)
        assert isinstance(self.y_val, np.ndarray)

        self.val_loader = DataLoader(
            NumpyWindowDataset(self.x_val, self.y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        Return train and validation DataLoaders and sequence lengths.

        Returns:
            Tuple[DataLoader, DataLoader, int, int]:
            train_loader, val_loader, input_chunk, output_chunk
        """
        assert self.train_loader is not None and self.val_loader is not None
        return self.train_loader, self.val_loader, self.input_chunk, self.output_chunk

    def get_test_windows(self) -> Tuple:
        """
        Create input/output windows for the test set.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x_test, y_test
        """
        return self.x_test, self.y_test
