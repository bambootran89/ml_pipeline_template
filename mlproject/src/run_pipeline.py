import pandas as pd

from mlproject.src.config_loader import load_config
from mlproject.src.data.dataloader import create_windows
from mlproject.src.data.dataset import TSDataset
from mlproject.src.data.dl_datamodule import DLDataModule
from mlproject.src.eval.evaluator import mae, mse, smape
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.trainer import train_model


def preprocess_data(cfg):
    """Run offline preprocessing and return train/val/test splits."""
    preprocessor = OfflinePreprocessor(cfg)
    df = preprocessor.run()
    dataset = TSDataset(df, cfg)
    return dataset.train(), dataset.val(), dataset.test()


def create_loaders(train_df, val_df, target_column, hyperparams, cfg):
    """
    Create windowed datasets and PyTorch DataLoaders using DLDataModule.

    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        target_column (str): Target column name
        hyperparams (dict):
        Model hyperparameters containing input/output chunk lengths and batch size
        cfg (dict): Config dict for training options

    Returns:
        Tuple[DataLoader, DataLoader, int, int]:
        train_loader, val_loader, input_chunk, output_chunk
    """
    input_chunk = int(hyperparams.get("input_chunk_length", 24))
    output_chunk = int(hyperparams.get("output_chunk_length", 6))
    batch_size = int(hyperparams.get("batch_size", 16))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    # Initialize DLDataModule with train+val+test concatenated (DLDataModule
    # tự chia split)
    dl_module = DLDataModule(
        df=pd.concat([train_df, val_df]),  # DLDataModule sẽ xử lý train/val/test
        cfg=cfg,
        target_column=target_column,
    )

    # Setup DataModule (windowing + DataLoader)
    dl_module.setup(
        input_chunk=input_chunk,
        output_chunk=output_chunk,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Get DataLoaders
    train_loader, val_loader, input_chunk, output_chunk = dl_module.get_loaders()

    return train_loader, val_loader, input_chunk, output_chunk


def initialize_model_wrapper(approach):
    """Return a model wrapper based on approach config."""
    model_name = approach.get("model")
    hyperparams = approach.get("hyperparams", {})
    if model_name == "nlinear":
        return NLinearWrapper(hyperparams)
    elif model_name == "tft":
        return TFTWrapper(hyperparams)
    else:
        raise RuntimeError(f"Unknown model: {model_name}")


def evaluate_model(wrapper, test_df, target_column, input_chunk, output_chunk):
    """Create test windows, predict, and print metrics."""
    x_test, y_test = create_windows(test_df, target_column, input_chunk, output_chunk)
    preds = wrapper.predict(x_test)
    print(
        f"Test metrics - MAE: {mae(y_test, preds):.6f}, "
        f"MSE: {mse(y_test, preds):.6f}, "
        f"SMAPE: {smape(y_test, preds):.6f}"
    )


def run_approach_pipeline(approach, train_df, val_df, test_df, target_column, cfg):
    """Complete pipeline for a single approach."""
    train_loader, val_loader, input_chunk, output_chunk = create_loaders(
        train_df, val_df, target_column, approach.get("hyperparams", {}), cfg
    )

    wrapper = initialize_model_wrapper(approach)
    wrapper = train_model(
        wrapper,
        train_loader,
        val_loader,
        approach.get("hyperparams", {}),
        device=cfg.get("training", {}).get("device", "cpu"),
        save_dir=cfg.get("training", {}).get(
            "artifacts_dir", "mlproject/artifacts/models"
        ),
    )

    evaluate_model(wrapper, test_df, target_column, input_chunk, output_chunk)


def main(cfg_path: str = ""):
    """
    Run the full ML pipeline: preprocessing, training, and evaluation.
    """
    cfg = load_config(cfg_path)

    # Preprocess and split data
    train_df, val_df, test_df = preprocess_data(cfg)

    # Select first target for demo
    target_column = cfg.get("data", {}).get("target_columns", ["HUFL"])[0]

    # Run all approaches
    for approach in cfg.get("experiment", {}).get("approaches", []):
        print("Running approach:", approach.get("name"))
        run_approach_pipeline(approach, train_df, val_df, test_df, target_column, cfg)


if __name__ == "__main__":
    main()
