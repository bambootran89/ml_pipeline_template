from torch.utils.data import DataLoader

from mlproject.src.datamodule.dataset import NumpyWindowDataset


class FoldDataLoaderBuilder:
    """
    Utility class for constructing train/validation DataLoaders
    used inside each cross-validation fold.

    This isolates the creation of PyTorch DataLoaders so the fold
    pipeline remains clean and modular.
    """

    @staticmethod
    def build(x_train, y_train, x_val, y_val, batch_size: int):
        """
        Create train and validation DataLoaders for a fold.

        Parameters
        ----------
        x_train : np.ndarray
            Training input features.
        y_train : np.ndarray
            Training target values.
        x_val : np.ndarray
            Validation/test input features.
        y_val : np.ndarray
            Validation/test target values.
        batch_size : int
            Batch size for both loaders.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            (train_loader, val_loader)
        """
        return (
            DataLoader(
                NumpyWindowDataset(x_train, y_train),
                batch_size=batch_size,
                shuffle=True,
            ),
            DataLoader(
                NumpyWindowDataset(x_val, y_val),
                batch_size=batch_size,
                shuffle=False,
            ),
        )
