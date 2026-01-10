# API Testing Examples with Realistic Data

This document provides realistic API testing examples using actual data structure from ETTh1 dataset.

## Data Structure

Based on `mlproject/configs/base/data.yaml` and `mlproject/configs/experiments/etth1.yaml`:

- **Features**: `["HUFL", "MUFL", "mobility_inflow"]`
- **Target columns**: `["HUFL", "MUFL"]`
- **Index column**: `"date"`
- **Input chunk length**: `24` (from `experiment.hyperparams.input_chunk_length`)
- **Output chunk length**: `6` (model predicts 6 future timesteps)

## Example 1: Minimal Test (cURL)

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

**Expected Response:**

```json
{
  "predictions": [5.628, 5.701, 5.823, 5.945, 6.078, 6.201]
}
```

## Example 2: Python Requests

```python
import requests

# Health check first
health_response = requests.get("http://localhost:8000/health")
print("Health:", health_response.json())

# Prepare prediction payload (24 timesteps required)
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

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)

print("Predictions:", response.json())
```

## Example 3: Using Helper Script

Generate test data automatically:

```python
from examples.generate_test_data import generate_test_data
import requests

# Generate 24 timesteps of test data
test_data = generate_test_data(num_timesteps=24)

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": test_data}
)

print(response.json())
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

## Common Errors

### Error: Insufficient Data

```json
{
  "detail": "Need at least 24 rows, got 10"
}
```

**Solution**: Provide exactly 24 timesteps (input_chunk_length)

### Error: Missing Features

```json
{
  "detail": "Missing required features: ['HUFL', 'MUFL', 'mobility_inflow']"
}
```

**Solution**: Include all three features in the data payload

### Error: Model Not Loaded

```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

**Solution**: Check MLflow connection and model availability

## Reference

- Data config: `mlproject/configs/base/data.yaml`
- Experiment config: `mlproject/configs/experiments/etth1.yaml`
- Helper script: `examples/generate_test_data.py`
- API documentation: `README_API.md`
