import argparse

from mlproject.src.pipeline.run_pipeline import main


def run():
    """Run the full ML pipeline."""
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="mlproject/configs/experiments/etth1.yaml",
        help="Path to experiment config YAML",
    )
    parser.parse_args()
    main()  # Không truyền cfg_path
