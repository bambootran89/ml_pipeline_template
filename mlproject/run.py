"""Top-level runner for mlproject package."""

import argparse

from mlproject.src.run_pipeline import main


def run():
    """Run the full ML pipeline."""
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="mlproject/configs/experiments/etth1.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    main(cfg_path=args.config)


if __name__ == "__main__":
    run()
