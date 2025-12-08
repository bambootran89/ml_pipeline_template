from mlproject.src.config_loader import load_config
from mlproject.src.datamodele.tsdl import TSDLDataModule
from mlproject.src.eval.evaluator import mae, mse, smape
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.trainer.trainer import train_model


def preprocess_data(cfg):
    """Run offline preprocessing and return train/val/test splits."""
    preprocessor = OfflinePreprocessor(cfg)
    df = preprocessor.run()
    return df


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


def evaluate_model(wrapper, x_test, y_test):
    """Create test windows, predict, and print metrics."""
    preds = wrapper.predict(x_test)
    print(
        f"Test metrics - MAE: {mae(y_test, preds):.6f}, "
        f"MSE: {mse(y_test, preds):.6f}, "
        f"SMAPE: {smape(y_test, preds):.6f}"
    )


def run_approach_pipeline(approach, history_df, target_column, cfg):
    """Complete pipeline for a single approach."""

    hyperparams = approach.get("hyperparams", {})
    input_chunk = int(hyperparams.get("input_chunk_length", 24))
    output_chunk = int(hyperparams.get("output_chunk_length", 6))
    batch_size = int(hyperparams.get("batch_size", 16))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    # Initialize DLDataModule with train+val+test concatenated (DLDataModule
    # tự chia split)
    dl_module = TSDLDataModule(
        df=history_df,  # DLDataModule sẽ xử lý train/val/test
        cfg=cfg,
        target_column=target_column,
        input_chunk=input_chunk,
        output_chunk=output_chunk,
    )

    # Setup DataModule (windowing + DataLoader)
    dl_module.setup(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Get DataLoaders
    train_loader, val_loader, _, _ = dl_module.get_loaders()

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

    x_test, y_test = dl_module.get_test_windows()

    evaluate_model(wrapper, x_test, y_test)


def main(cfg_path: str = ""):
    """
    Run the full ML pipeline: preprocessing, training, and evaluation.
    """
    cfg = load_config(cfg_path)

    # Preprocess and split data
    history_df = preprocess_data(cfg)

    # Select first target for demo
    target_column = cfg.get("data", {}).get("target_columns", ["HUFL"])[0]

    # Run all approaches
    for approach in cfg.get("experiment", {}).get("approaches", []):
        print("Running approach:", approach.get("name"))
        run_approach_pipeline(approach, history_df, target_column, cfg)


if __name__ == "__main__":
    main()
