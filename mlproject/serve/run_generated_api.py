#!/usr/bin/env python3
"""Auto-generate and run serve API from config.

This script automatically generates API code from serve configuration
and runs it immediately. No manual steps needed!

Usage:
    python -m mlproject.serve.run_generated_api \\
        --serve-config mlproject/configs/generated/standard_train_serve.yaml \\
        --framework fastapi \\
        --port 8000

    # Or with all options:
    python -m mlproject.serve.run_generated_api \\
        --serve-config mlproject/configs/generated/standard_train_serve.yaml \\
        --train-config mlproject/configs/pipelines/standard_train.yaml \\
        --framework ray \\
        --port 8000 \\
        --host 0.0.0.0
"""

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

from mlproject.src.utils.generator.config_generator import ConfigGenerator


def generate_and_run(
    serve_config_path: str,
    train_config_path: str,
    framework: str = "fastapi",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Generate API code and run it immediately.

    Args:
        serve_config_path: Path to serve YAML config.
        train_config_path: Path to training YAML config.
        framework: 'fastapi' or 'ray'.
        host: Host to bind to.
        port: Port to bind to.
    """
    print("=" * 60)
    print(f"Auto-Generate & Run {framework.upper()} API")
    print("=" * 60)
    print(f"Serve config: {serve_config_path}")
    print(f"Train config: {train_config_path}")
    print(f"Framework: {framework}")
    print(f"Address: {host}:{port}")
    print("=" * 60)

    # Step 1: Generate API code to persistent directory
    print("\n[1/2] Generating API code...")

    output_dir = "mlproject/serve/generated"
    generator = ConfigGenerator(train_config_path)

    try:
        api_path = generator.generate_api(
            serve_config_path=serve_config_path,
            output_dir=output_dir,
            framework=framework,
            experiment_config_path=train_config_path,
        )
        print(f"Generated: {api_path}")
    except Exception as e:
        print(f"Generation failed: {e}")
        sys.exit(1)

    # Step 2: Modify generated code to use custom host/port
    print("\n[2/3] Configuring server settings...")

    with open(api_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Inject host/port for FastAPI
    if framework == "fastapi":
        old_uvicorn_call = (
            'uvicorn.run(\n        "mlproject.serve.api:app",\n        '
            'host="0.0.0.0",\n        port=8000,'
        )
        new_uvicorn_call = (
            f'uvicorn.run(\n        app,\n        host="{host}",\n        '
            f'port={port},'
        )
        code = code.replace(old_uvicorn_call, new_uvicorn_call)
        code = code.replace(
            'uvicorn.run(app, host="0.0.0.0", port=8000)',
            f'uvicorn.run(app, host="{host}", port={port})',
        )

    with open(api_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Configured: {host}:{port}")

    # Step 3: Run the generated API
    print(f"\n[3/3] Starting {framework.upper()} server...")
    print(f"\n{'='*60}")
    print(f"API starting at: http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print(f"{'='*60}")
    print("\nPress Ctrl+C to stop the server\n")
    print("-" * 60)

    try:
        # Run the generated file
        subprocess.run([sys.executable, api_path], check=False)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Server stopped by user")
        print("=" * 60)
    except Exception as e:
        print(f"\nServer failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-generate and run serve API from config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FastAPI on port 8000
  python -m mlproject.serve.run_generated_api \\
      --serve-config mlproject/configs/generated/standard_train_serve.yaml \\
      --framework fastapi

  # Ray Serve on port 9000
  python -m mlproject.serve.run_generated_api \\
      --serve-config mlproject/configs/generated/standard_train_serve.yaml \\
      --framework ray \\
      --port 9000

  # With custom train config
  python -m mlproject.serve.run_generated_api \\
      --serve-config mlproject/configs/generated/conditional_branch_serve.yaml \\
      --train-config mlproject/configs/pipelines/conditional_branch.yaml \\
      --framework fastapi
        """,
    )

    parser.add_argument(
        "--serve-config",
        required=True,
        help="Path to serve YAML config (e.g., standard_train_serve.yaml)",
    )
    parser.add_argument(
        "--train-config",
        help="Path to training YAML config (auto-inferred if not provided)",
    )
    parser.add_argument(
        "--framework",
        choices=["fastapi", "ray"],
        default="fastapi",
        help="Framework to use (default: fastapi)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    args = parser.parse_args()

    # Auto-infer train config if not provided
    train_config = args.train_config
    if not train_config:
        # Try to infer from serve config name
        # e.g., standard_train_serve.yaml -> standard_train.yaml
        serve_path = Path(args.serve_config)
        train_name = serve_path.stem.replace("_serve", "")
        train_config = f"mlproject/configs/pipelines/{train_name}.yaml"

        if not Path(train_config).exists():
            print(f"Could not auto-infer train config from: {args.serve_config}")
            print("  Please provide --train-config explicitly")
            sys.exit(1)

        print(f"Auto-inferred train config: {train_config}")

    # Generate and run
    generate_and_run(
        serve_config_path=args.serve_config,
        train_config_path=train_config,
        framework=args.framework,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
