import numpy as np
from torch.utils.data import DataLoader, Dataset


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


def create_windows(df, target_col, input_chunk, output_chunk, stride=1):
    """
    Convert a DataFrame into input/output windows for training.

    Args:
        df (pd.DataFrame): Input dataframe with features and target.
        target_col (str): Target column name.
        input_chunk (int): Length of input sequence.
        output_chunk (int): Length of output sequence.
        stride (int): Sliding window step.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
         x_windows (N, seq_len, feat), y_windows (N, horizon)
    """
    features = df.columns.tolist()
    n = len(df)
    x_windows, y_windows = [], []

    for end_idx in range(input_chunk, n - output_chunk + 1, stride):
        start_idx = end_idx - input_chunk
        x_window = df.iloc[start_idx:end_idx].values  # (seq_len, feat)
        y_window = df.iloc[end_idx : end_idx + output_chunk][
            target_col
        ].values  # (output_chunk,)
        x_windows.append(x_window)
        y_windows.append(y_window)

    if len(x_windows) == 0:
        return np.zeros((0, input_chunk, len(features))), np.zeros((0, output_chunk))

    return np.stack(x_windows), np.stack(y_windows)


def get_dataloaders(
    x_train_array, y_train_array, x_val_array, y_val_array, batch_size=16, num_workers=0
):
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        x_train_array (np.ndarray): Training features (N, seq_len, feat)
        y_train_array (np.ndarray): Training targets (N, horizon)
        x_val_array (np.ndarray): Validation features (M, seq_len, feat)
        y_val_array (np.ndarray): Validation targets (M, horizon)
        batch_size (int, optional): Batch size. Defaults to 16.
        num_workers (int, optional): Number of DataLoader workers. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, val_loader
    """
    train_dataset = NumpyWindowDataset(x_train_array, y_train_array)
    val_dataset = NumpyWindowDataset(x_val_array, y_val_array)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
