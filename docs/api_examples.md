# API Testing Examples

Realistic API testing examples using ETTh1 dataset structure from experiment configs.

## Configuration

From `mlproject/configs/experiments/etth1.yaml` and `mlproject/configs/base/data.yaml`:

```yaml
# Data configuration
data:
  features: ["HUFL", "MUFL", "mobility_inflow"]
  target_columns: ["HUFL", "MUFL"]
  index_col: "date"

# Model configuration
experiment:
  hyperparams:
    input_chunk_length: 24    # Required input timesteps
    output_chunk_length: 6    # Predicted future timesteps
    n_features: 3             # Number of input features
    n_targets: 2              # Number of prediction targets
```

**Prediction Output:**
- Model predicts `n_targets * output_chunk_length` values
- For ETTh1: 2 targets × 6 timesteps = 12 prediction values
- Format: Flattened array of all predictions

## Example 1: Complete Request (cURL)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": [
        "2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00",
        "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00",
        "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00",
        "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00",
        "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00",
        "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00",
        "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00",
        "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"
      ],
      "HUFL": [
        5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763,
        9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745,
        6.233, 5.838, 5.582, 5.465, 5.465, 5.557
      ],
      "MUFL": [
        1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896,
        2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125,
        3.067, 2.969, 2.838, 2.683, 2.515, 2.346
      ],
      "mobility_inflow": [
        1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234,
        3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567,
        4.678, 4.789, 4.890, 4.901, 4.912, 4.923
      ]
    }
  }'
```


## Example 2: Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# Prediction request (24 timesteps required)
payload = {
    "data": {
        "date": [
            "2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00",
            "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00",
            "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00",
            "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00",
            "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00",
            "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00",
            "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00",
            "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"
        ],
        "HUFL": [
            5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763,
            9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745,
            6.233, 5.838, 5.582, 5.465, 5.465, 5.557
        ],
        "MUFL": [
            1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896,
            2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125,
            3.067, 2.969, 2.838, 2.683, 2.515, 2.346
        ],
        "mobility_inflow": [
            1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234,
            3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567,
            4.678, 4.789, 4.890, 4.901, 4.912, 4.923
        ]
    }
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()

print("Predictions:", result["predictions"])
print("Number of predictions:", len(result["predictions"]))
```

## Example 3: Using Helper Script

```python
from examples.generate_test_data import generate_test_data
import requests

# Generate test data (24 timesteps)
test_data = generate_test_data(num_timesteps=24)

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": test_data}
)

result = response.json()
print(f"Received {len(result['predictions'])} predictions")
print(result)
```

## Example 4: Load Testing

Create `test_payload.json`:

```json
{
  "data": {
    "date": [
      "2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00",
      "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00",
      "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00",
      "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00",
      "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00",
      "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00",
      "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00",
      "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"
    ],
    "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
    "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
    "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
  }
}
```

Run load test:

```bash
# Install apache bench
sudo apt-get install apache2-utils

# 1000 requests, 10 concurrent
ab -n 1000 -c 10 -p test_payload.json -T application/json \
   http://localhost:8000/predict
```

## Understanding Output Format

The number of predictions depends on model configuration:

```
predictions_count = n_targets × output_chunk_length
```

**For ETTh1 config:**
- n_targets = 2 (HUFL, MUFL)
- output_chunk_length = 6
- predictions_count = 2 × 6 = 12

**Output format (flattened):**
```python
[
  # HUFL predictions (6 timesteps)
  pred_HUFL_t1, pred_HUFL_t2, pred_HUFL_t3, pred_HUFL_t4, pred_HUFL_t5, pred_HUFL_t6,
  # MUFL predictions (6 timesteps)
  pred_MUFL_t1, pred_MUFL_t2, pred_MUFL_t3, pred_MUFL_t4, pred_MUFL_t5, pred_MUFL_t6
]
```

## Configuration Reference

All values are read from configs (no hardcoding):

| Parameter | Config Location | ETTh1 Value |
|-----------|----------------|-------------|
| input_chunk_length | experiment.hyperparams | 24 |
| output_chunk_length | experiment.hyperparams | 6 |
| n_features | experiment.hyperparams | 3 |
| n_targets | experiment.hyperparams | 2 |
| features | data.features | ["HUFL", "MUFL", "mobility_inflow"] |
| target_columns | data.target_columns | ["HUFL", "MUFL"] |

## See Also

- Experiment config: `mlproject/configs/experiments/etth1.yaml`
- Data config: `mlproject/configs/base/data.yaml`
- Helper script: `examples/generate_test_data.py`
- API documentation: `docs/readme_api.md`
