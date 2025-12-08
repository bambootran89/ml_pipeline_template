# simple fallback NLinear-like model implemented with linear layers
import torch
from torch import nn

from mlproject.src.models.base import ModelWrapperBase


class FallbackNLinear(nn.Module):
    """
    Simple feedforward neural network with one hidden layer and ReLU activation.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        hidden (int, optional): Number of hidden units. Defaults to 128.
    """

    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim) or
                              (batch, seq_len, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch, output_dim).
        """
        # x shape: (batch, seq_len, feat) or (batch, input_dim)
        if x.dim() == 3:
            b, s, f = x.shape
            x = x.reshape(b, s * f)
        return self.net(x)


class NLinearWrapper(ModelWrapperBase):
    """
    Wrapper class for FallbackNLinear providing build,
    training, and prediction methods.

    Args:
        cfg (dict, optional): Configuration dictionary.
        Can include 'hidden' key for hidden units.
    """

    def build(self, input_dim, output_dim):
        """
        Build the FallbackNLinear model with specified input and output dimensions.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
        """
        hidden = self.cfg.get("hidden", 128)
        self.model = FallbackNLinear(input_dim, output_dim, hidden=hidden)

    def train_step(self, batch, optimizer, loss_fn, device):
        """
        Perform a single training step (forward, loss, backward, optimizer step).

        Args:
            batch (tuple): Tuple (x, y) of training data as numpy arrays.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            loss_fn (callable): Loss function.
            device (torch.device): Device to run the computation on.

        Returns:
            float: Computed loss value for the batch.
        """
        self.model.train()
        x, y = batch
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        preds = self.model(x)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x_numpy):
        """
        Predict output for given input numpy array.

        Args:
            x_numpy (np.ndarray): Input data array of shape (batch, input_dim).

        Returns:
            np.ndarray: Predicted outputs as numpy array.
        """
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x_numpy.astype("float32"))
            out = self.model(t)
            return out.numpy()
