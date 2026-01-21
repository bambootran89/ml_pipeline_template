#!/bin/bash
# test_api.sh - Comprehensive API Verification Script
# Usage: ./test_api.sh [PORT default:8000]

PORT=${1:-8000}
HOST="localhost"
BASE_URL="http://$HOST:$PORT"

echo "=================================================="
echo " Starting API Verification on $BASE_URL"
echo "=================================================="

# Helper function for JSON parsing
parse_json() {
    python3 -c "import sys, json; print(json.load(sys.stdin))" 2>/dev/null || echo "Error parsing JSON response"
}

# --------------------------------------------------------
# 1. Health Check & Type Detection
# --------------------------------------------------------
echo -e "\n[1/4] Testing Health Check (/health)..."
HEALTH_JSON=$(curl -s "$BASE_URL/health")
echo "$HEALTH_JSON" | parse_json

DATA_TYPE=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('data_type', 'unknown'))" 2>/dev/null)
echo "Detected Data Type: $DATA_TYPE"

# --------------------------------------------------------
# 2. Standard Predict (Common)
# --------------------------------------------------------
echo -e "\n[2/4] Testing Standard Predict (/predict)..."

if [ "$DATA_TYPE" == "tabular" ]; then
    PAYLOAD='{
        "data": {
          "Pclass": [3],
          "Age": [22.0],
          "SibSp": [1],
          "Parch": [0],
          "Fare": [7.25],
          "Sex": ["male"],
          "Embarked": ["S"]
        }
    }'
else
    PAYLOAD='{
        "data": {
          "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
          "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
          "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
          "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
        }
    }'
fi

curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | parse_json

# --------------------------------------------------------
# 3. Batch/Feast Prediction (Conditional)
# --------------------------------------------------------
if [ "$DATA_TYPE" == "tabular" ]; then
    echo -e "\n[3/4] Testing Tabular Batch Predict (/predict/batch)..."
    curl -s -X POST "$BASE_URL/predict/batch" \
      -H "Content-Type: application/json" \
      -d '{
        "data": {
          "Pclass": [3, 1, 3, 1, 3],
          "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
          "SibSp": [1, 1, 0, 1, 0],
          "Parch": [0, 0, 0, 0, 0],
          "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
          "Sex": ["male", "female", "female", "female", "male"],
          "Embarked": ["S", "C", "S", "S", "S"]
        }
      }' | parse_json
elif [ "$DATA_TYPE" == "feast" ]; then
    echo -e "\n[3/4] Testing Feast Batch Predict (/predict/feast/batch)..."
    curl -s -X POST "$BASE_URL/predict/feast/batch" \
      -H "Content-Type: application/json" \
      -d '{
        "time_point": "2024-01-09T00:00:00",
        "entities": [1, 2, 3]
      }' | parse_json
else
    echo -e "\n[3/4] Skipped Batch Predict (Time Series Mode)"
fi

# --------------------------------------------------------
# 4. Multistep Prediction (Conditional)
# --------------------------------------------------------
if [ "$DATA_TYPE" == "timeseries" ]; then
    echo -e "\n[4/4] Testing Multistep Predict (/predict/multistep)..."
    curl -s -X POST "$BASE_URL/predict/multistep" \
      -H "Content-Type: application/json" \
      -d '{
        "steps_ahead": 12,
        "data": {
          "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
          "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
          "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
          "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
        }
      }' | parse_json
else
    echo -e "\n[4/4] Skipped Multistep Predict (Tabular Mode)"
fi

echo -e "\n=================================================="
echo " Tests Complete."
echo "=================================================="
