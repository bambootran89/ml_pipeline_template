import os

import numpy as np
import torch


def infer_dims_from_batch(xb, yb):
    """Infer input and output dimensions from a batch."""
    if hasattr(xb, "shape"):
        input_dim = xb.shape[1] * xb.shape[2] if len(xb.shape) == 3 else xb.shape[1]
        output_dim = yb.shape[1] if len(yb.shape) > 1 else 1
    else:
        input_dim = xb[0].shape[0]
        output_dim = yb.shape[1]
    return input_dim, output_dim


def setup_optimizer(wrapper, lr):
    """Return Adam optimizer for wrapper.model."""
    return torch.optim.Adam(wrapper.model.parameters(), lr=lr)


def train_one_epoch(wrapper, train_loader, optimizer, loss_fn, device):
    """Run one training epoch and return mean loss."""
    losses = []
    for xb, yb in train_loader:
        x_batch = xb.numpy() if hasattr(xb, "numpy") else xb
        y_batch = yb.numpy() if hasattr(yb, "numpy") else yb
        loss = wrapper.train_step((x_batch, y_batch), optimizer, loss_fn, device)
        losses.append(loss)
    return np.mean(losses)


def validate_model(wrapper, val_loader):
    """Compute validation MSE for one epoch."""
    losses = []
    for xb, yb in val_loader:
        x_batch = xb.numpy() if hasattr(xb, "numpy") else xb
        y_true = yb.numpy() if hasattr(yb, "numpy") else yb
        preds = wrapper.predict(x_batch)
        losses.append(np.mean((preds - y_true) ** 2))
    return np.mean(losses)


def save_model(wrapper, save_dir):
    """Save the model state_dict if it exists."""
    os.makedirs(save_dir, exist_ok=True)
    if hasattr(wrapper.model, "state_dict"):
        torch.save(wrapper.model.state_dict(), os.path.join(save_dir, "model.pt"))


def train_model(
    wrapper,
    train_loader,
    val_loader,
    hyperparams,
    device="cpu",
    save_dir="mlproject/artifacts/models",
):
    """
    Orchestrates training using helper functions.
    """
    # --- setup ---
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    n_epochs = int(hyperparams.get("n_epochs", 1))
    # --- build model if needed ---
    if wrapper.model is None:
        xb, yb = next(iter(train_loader))
        input_dim, output_dim = infer_dims_from_batch(xb, yb)
        wrapper.build(input_dim, output_dim)

    wrapper.model.to(device)
    optimizer = setup_optimizer(wrapper, float(hyperparams.get("lr", 1e-3)))

    # --- training loop ---
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            wrapper, train_loader, optimizer, torch.nn.MSELoss(), device
        )
        val_loss = validate_model(wrapper, val_loader)
        print(
            f"Epoch {epoch}/{n_epochs} \
            - train_loss={train_loss:.6f} val_mse={val_loss:.6f}"
        )

    # --- save model ---
    save_model(wrapper, save_dir)

    return wrapper
