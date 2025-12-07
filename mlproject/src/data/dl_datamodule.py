from typing import Optional, Tuple

from torch.utils.data import DataLoader

from mlproject.src.data.base_datamodule import BaseDataModule
from mlproject.src.data.dataloader import NumpyWindowDataset, create_windows


class DLDataModule(BaseDataModule):
    """
    DL DataModule for PyTorch models.
    Creates windowed dataset + DataLoader using dataloader.py utilities.
    """

    def __init__(self, df, cfg, target_column: str):
        """
        Initialize DLDataModule.

        Args:
            df (pd.DataFrame): Full dataset
            cfg (dict): Configuration dict
            target_column (str): Name of target column
        """
        super().__init__(df, cfg, target_column)
        self.input_chunk: Optional[int] = None
        self.output_chunk: Optional[int] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def setup(
        self,
        input_chunk: int,
        output_chunk: int,
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
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk

        x_train, y_train = create_windows(
            self.train_df,
            self.target_column,
            input_chunk,
            output_chunk,
        )
        x_val, y_val = create_windows(
            self.val_df,
            self.target_column,
            input_chunk,
            output_chunk,
        )

        self.train_loader = DataLoader(
            NumpyWindowDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            NumpyWindowDataset(x_val, y_val),
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
        assert self.input_chunk is not None and self.output_chunk is not None
        return self.train_loader, self.val_loader, self.input_chunk, self.output_chunk

    def get_test_windows(self) -> Tuple:
        """
        Create input/output windows for the test set.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x_test, y_test
        """
        assert self.input_chunk is not None and self.output_chunk is not None
        x_test, y_test = create_windows(
            self.test_df, self.target_column, self.input_chunk, self.output_chunk
        )
        return x_test, y_test
