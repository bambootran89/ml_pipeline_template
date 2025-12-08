import numpy as np
from torch.utils.data import Dataset


class NumpyWindowDataset(Dataset):
    """
    PyTorch Dataset wrapper for numpy arrays.

    Stores features X (N, seq_len, feat) and targets y (N, horizon).
    """

    def __init__(self, x_array: np.ndarray, y_array: np.ndarray):
        """
        Args:
            x_array (np.ndarray): Input features of shape (N, seq_len, feat)
            y_array (np.ndarray): Targets of shape (N, horizon)
        """
        self.x_array = x_array.astype("float32")
        self.y_array = y_array.astype("float32")

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        return self.x_array[idx], self.y_array[idx]
