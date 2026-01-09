# Conditional Branch Fix for Eval/Serve Modes

## Problem

When using conditional branches in ML pipelines, a critical issue occurred when the condition evaluated differently between training and serving:

### Scenario
1. **Training Phase**:
   - Condition: `data_size > 100` evaluates to `TRUE`
   - Only `train_tft` model is trained and saved to MLflow
   - `train_xgb` model is never trained

2. **Serving Phase**:
   - Condition: `data_size > 100` evaluates to `FALSE` (different data!)
   - Pipeline tries to use `train_xgb` model
   - **ERROR**: `KeyError: Step 'train_xgb_inference': Required input 'model' not found at key 'train_xgb_model'`

### Root Cause

Conditional branches are **data-dependent**, meaning:
- Training data and serving data may have different characteristics
- The condition can evaluate to different values in different phases
- Only ONE model is trained during training (based on the condition)
- But eval/serve tries to use whichever model the condition selects at runtime

This creates a mismatch when:
```
Training: condition=TRUE  → train_tft model saved
Serving:  condition=FALSE → try to use train_xgb model → FAIL (doesn't exist)
```

## Solution

Added intelligent **fallback logic** to `BranchStep` in `mlproject/src/pipeline/steps/advanced_step.py`:

### How It Works

1. **Evaluate Condition**: Determine which branch should be executed (primary)

2. **Check Model Availability**: Before executing, check if the required model exists in context
   - For `inference` and `evaluator` steps, verify the model key exists
   - Uses new `_check_model_availability()` method

3. **Execute or Fallback**:
   - If primary branch has its model → Execute primary branch
   - If primary branch missing model → Try fallback branch (the other branch)
   - If neither branch has a model → Raise clear error with guidance

4. **Helpful Error Message**: If both branches fail, provide actionable guidance:
   ```
   Step 'model_selection': Cannot execute conditional branch.
   Neither branch has its required model available in context.
   Available model keys: ['train_tft_model', 'preprocess_model'].
   This typically happens when the condition evaluates differently
   between training and serving. Consider training both models
   separately before using conditional branching for serving.
   ```

### Example Flow

```
Serving Pipeline Execution:
├─ init_artifacts (MLflow Loader)
│  ├─ Load train_tft_model: ✓ SUCCESS
│  └─ Load train_xgb_model: ✗ WARNING (not found, skipped)
│
├─ preprocess
│  └─ Output: preprocessed_data
│
├─ model_selection (Branch with NEW FALLBACK LOGIC)
│  ├─ Condition: data_size > 100 = FALSE
│  ├─ Primary: train_xgb_inference (if_false)
│  │  └─ Check: train_xgb_model in context? NO
│  │     └─ ⚠️  Primary model not available
│  │
│  ├─ Fallback: train_tft_inference (if_true)
│  │  └─ Check: train_tft_model in context? YES
│  │     └─ ✓ Execute train_tft_inference
│  │
│  └─ Result: train_tft_predictions
│
└─ final_profiling
   └─ Profile results
```

## Code Changes

### Added Method: `_check_model_availability()`

```python
def _check_model_availability(
    self, branch_config: Dict[str, Any], context: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """Check if required model exists in context for inference/evaluator steps.

    Returns:
        (available, model_key): True if model is available or not required,
                               False if model is required but missing.
    """
    if not branch_config:
        return True, None

    step_type = branch_config.get("type", "")
    if step_type not in ["inference", "evaluator"]:
        return True, None

    # Check if step has wiring with model input
    wiring = branch_config.get("wiring", {})
    inputs = wiring.get("inputs", {})
    model_key = inputs.get("model")

    if model_key and model_key not in context:
        return False, model_key

    return True, model_key
```

### Modified Method: `execute()`

The execute method now:
1. Determines primary and fallback branches based on condition
2. Checks model availability for primary branch
3. Falls back to other branch if primary model is missing
4. Raises helpful error if neither branch can execute

## Benefits

1. **Robustness**: Pipeline works even when condition evaluates differently
2. **Automatic Recovery**: Intelligently selects available model
3. **Clear Errors**: Provides actionable guidance when both models are missing
4. **Backward Compatible**: Doesn't break existing pipelines that work correctly

## Usage Notes

### When This Fix Helps

- You trained a pipeline with conditional branch that trained only ONE model
- You want to use the pipeline for serving/evaluation with different data
- The serving data characteristics differ from training data
- The condition evaluates to a different value during serving

### Best Practices

1. **Single Model Training**: If you only need one model, train with conditional branch as normal. The fallback will ensure the trained model is used during serving.

2. **Multi-Model Training**: If you need both models available:
   - Train both models separately (without conditional branch in training)
   - Use conditional branch only in serving to select the appropriate model

3. **Consistent Conditions**: If possible, ensure serving data has similar characteristics to training data so conditions evaluate consistently

## Testing

To verify the fix works:

```bash
# 1. Generate serve config
python -m mlproject.src.pipeline.dag_run generate \
  -t mlproject/configs/pipelines/conditional_branch.yaml \
  -o mlproject/configs/generated \
  -a latest \
  --type serve

# 2. Run serving (will use fallback if condition differs from training)
python -m mlproject.src.pipeline.dag_run serve \
  -e mlproject/configs/experiments/etth3.yaml \
  -p mlproject/configs/generated/conditional_branch_serve.yaml \
  -i ./sample_input.csv \
  -a latest
```

Expected output should show:
```
[model_selection] Condition: data_size > 100 (actual=50) -> False
[model_selection] FALSE branch requires model 'train_xgb_model' which is not available in context
[model_selection] Falling back to TRUE branch (primary branch model not available)
[model_selection] Executing TRUE branch
[train_tft_inference] Running model.predict()
[train_tft_inference] Inference completed
```

## Related Files

- `mlproject/src/pipeline/steps/advanced_step.py`: Contains the BranchStep implementation
- `mlproject/src/pipeline/steps/mlflow_loader_step.py`: MLflow loader that gracefully handles missing models
- `mlproject/src/utils/config_generator.py`: Generates eval/serve configs from training configs
