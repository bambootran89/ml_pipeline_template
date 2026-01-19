# Manual Patch Instructions for nested_suppipeline_serve_ray.py

## Problem
The generated file still calls `predict_timeseries_multistep()` directly, bypassing feature generators.

## Fix Location 1: /predict/multistep endpoint (around line 525-545)

**FIND THIS CODE:**
```python
@app.post("/predict/multistep", response_model=MultiPredictResponse)
async def predict_multistep(
    self,
    request: BatchPredictRequest,
    steps_ahead: int = 6
) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = (
            await self.preprocess_handle.preprocess.remote(df)
        )
        context = {"preprocessed_data": preprocessed_data}
        result = await self.model_handle.predict_timeseries_multistep.remote(
            context, steps_ahead
        )
        return MultiPredictResponse(
            predictions=result["predictions"],
            metadata=result["metadata"]
        )
```

**REPLACE WITH:**
```python
@app.post("/predict/multistep", response_model=MultiPredictResponse)
async def predict_multistep(
    self,
    request: BatchPredictRequest,
    steps_ahead: int = 6
) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < self.INPUT_CHUNK_LENGTH:
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
            metadata=result.get("metadata")
        )
```

## Key Changes:
1. Added input length validation
2. Changed from `predict_timeseries_multistep.remote(context, steps_ahead)`
   to `run_full_pipeline.remote(preprocessed_data)`
3. Changed from `result["predictions"]` to `result.get("predictions", {})`

## Why This Fixes the Error:
- `run_full_pipeline()` calls `generate_additional_features()` and `compose_features()`
- This ensures cluster_1_features and pca_data are composed with base features
- Model receives 144 features (72 base + 72 additional) instead of just 72
