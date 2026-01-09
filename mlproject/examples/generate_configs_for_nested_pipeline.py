#!/usr/bin/env python3
"""
Example: Generate eval and serve configs for nested sub-pipeline

This script demonstrates how to use ConfigGenerator to automatically
generate evaluation and serving configurations from a training pipeline
that contains nested sub-pipelines.

Usage:
    python mlproject/examples/generate_configs_for_nested_pipeline.py
"""

from pathlib import Path
from mlproject.src.utils.config_generator import ConfigGenerator


def main():
    """Generate eval and serve configs for nested_suppipeline.yaml"""

    print("=" * 80)
    print("ConfigGenerator Example: Nested Sub-Pipeline Support")
    print("=" * 80)

    # Input: Training pipeline config with nested sub-pipeline
    train_config = "mlproject/configs/pipelines/nested_suppipeline.yaml"

    # Output directory for generated configs
    output_dir = "mlproject/configs/generated"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nInput: {train_config}")
    print(f"Output directory: {output_dir}\n")

    # Initialize ConfigGenerator
    generator = ConfigGenerator(train_config)

    # Method 1: Generate individual configs
    print("Method 1: Generate configs individually")
    print("-" * 80)

    eval_path = f"{output_dir}/nested_suppipeline_eval.yaml"
    generator.generate_eval_config(alias="latest", output_path=eval_path)

    serve_path = f"{output_dir}/nested_suppipeline_serve.yaml"
    generator.generate_serve_config(alias="latest", output_path=serve_path)

    # Method 2: Generate all configs at once
    print("\nMethod 2: Generate all configs at once")
    print("-" * 80)

    results = generator.generate_all(output_dir=output_dir, alias="production")
    for mode, path in results.items():
        print(f"  {mode}: {path}")

    # Display what was transformed
    print("\n" + "=" * 80)
    print("Transformation Summary")
    print("=" * 80)
    print("""
The ConfigGenerator automatically:

1. **Extracted nested model producers and preprocessors**
   - Found 'normalize' preprocessor inside feature_pipeline sub-pipeline
   - Found 'cluster' model producer inside feature_pipeline sub-pipeline
   - Found 'train_model' model producer at top level

2. **Generated eval config with:**
   - MLflow loader to restore all artifacts (normalize, cluster, train_model)
   - Transformed sub-pipeline with:
     * normalize: is_train=false, instance_key=normalize_model
     * cluster: wired to use loaded model
   - Evaluators for each model producer (cluster, train_model)
   - Profiling and logging steps

3. **Generated serve config with:**
   - MLflow loader to restore all artifacts
   - Transformed sub-pipeline for inference mode
   - Inference step for final model (train_model)
   - Profiling step

Key Features:
- ✓ Recursively extracts artifacts from nested sub-pipelines
- ✓ Preserves sub-pipeline structure (no flattening)
- ✓ Transforms internal steps for eval/serve mode
- ✓ Generates correct wiring and dependencies
- ✓ Supports multiple levels of nesting
    """)

    print("\nNext Steps:")
    print("  1. Review generated configs: " + eval_path)
    print("  2. Test eval pipeline:")
    print(f"     python -m mlproject.src.pipeline.dag_run eval \\")
    print(f"       -e <experiment_config> \\")
    print(f"       -p {eval_path} \\")
    print(f"       -a latest")
    print("  3. Test serve pipeline:")
    print(f"     python -m mlproject.src.pipeline.dag_run serve \\")
    print(f"       -e <experiment_config> \\")
    print(f"       -p {serve_path} \\")
    print(f"       -i <input_file> -a latest")
    print()


if __name__ == "__main__":
    main()
