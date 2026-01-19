#!/usr/bin/env python3
"""Auto-fix generated Ray Serve file to use run_full_pipeline."""

import re
import sys
from pathlib import Path

def fix_ray_file(file_path: str) -> bool:
    """Fix Ray Serve generated file to call run_full_pipeline.

    Args:
        file_path: Path to generated Ray serve file

    Returns:
        True if fixed, False if not needed
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return False

    content = path.read_text()

    # Check if already fixed
    if "run_full_pipeline.remote" in content:
        print(f"✅ File already patched: {file_path}")
        return True

    original = content

    # Fix 1: /predict/multistep endpoint
    old_pattern_multistep = r'''(\s+@app\.post\("/predict/multistep".*?\n\s+async def predict_multistep\(.*?\n.*?try:\n\s+df = pd\.DataFrame\(request\.data\)\n\s+)preprocessed_data = \(\n\s+await self\.preprocess_handle\.preprocess\.remote\(df\)\n\s+\)\n\s+context = \{"preprocessed_data": preprocessed_data\}\n\s+result = await self\.model_handle\.predict_timeseries_multistep\.remote\(\n\s+context, steps_ahead\n\s+\)\n\s+return MultiPredictResponse\(\n\s+predictions=result\["predictions"\],\n\s+metadata=result\["metadata"\]'''

    new_text_multistep = r'''\1if len(df) < self.INPUT_CHUNK_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input must have at least "
                        f"{self.INPUT_CHUNK_LENGTH} timesteps "
                        f"(got {len(df)})"
                    )
                )
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            # Use run_full_pipeline to ensure feature generators are called
            result = await self.model_handle.run_full_pipeline.remote(
                preprocessed_data
            )
            return MultiPredictResponse(
                predictions=result.get("predictions", {}),
                metadata=result.get("metadata")'''

    content = re.sub(old_pattern_multistep, new_text_multistep, content, flags=re.DOTALL)

    # Fix 2: /predict/batch endpoint (if tabular)
    old_pattern_batch = r'''(\s+@app\.post\("/predict/batch".*?\n\s+async def predict_batch\(.*?\n.*?try:\n\s+df = pd\.DataFrame\(request\.data\)\n\s+)preprocessed_data = \(\n\s+await self\.preprocess_handle\.preprocess\.remote\(df\)\n\s+\)\n\s+context = \{"preprocessed_data": preprocessed_data\}\n\s+result = await self\.model_handle\.predict_tabular_batch\.remote\(\n\s+context, request\.return_probabilities\n\s+\)'''

    new_text_batch = r'''\1preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            # Use run_full_pipeline to ensure feature generators are called
            result = await self.model_handle.run_full_pipeline.remote(
                preprocessed_data
            )'''

    content = re.sub(old_pattern_batch, new_text_batch, content, flags=re.DOTALL)

    if content != original:
        # Backup original
        backup_path = path.with_suffix('.py.backup')
        backup_path.write_text(original)
        print(f"📦 Backup created: {backup_path}")

        # Write fixed content
        path.write_text(content)
        print(f"✅ Fixed: {file_path}")
        print("\nChanges:")
        print("  - /predict/multistep now calls run_full_pipeline()")
        print("  - Added input length validation")
        print("  - Feature generators will now be executed")
        return True
    else:
        print(f"⚠️  No changes needed or pattern not found: {file_path}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_ray_generated.py <path_to_ray_file>")
        print("\nExample:")
        print("  python fix_ray_generated.py mlproject/serve/generated/nested_suppipeline_serve_ray.py")
        sys.exit(1)

    file_path = sys.argv[1]
    success = fix_ray_file(file_path)
    sys.exit(0 if success else 1)
