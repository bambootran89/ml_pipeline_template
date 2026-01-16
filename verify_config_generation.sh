#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Setup environment
export PYTHONPATH=.
if [ -d "mlproject_env" ]; then
    source mlproject_env/bin/activate
fi

PIPELINE_DIR="mlproject/configs/pipelines"
EXP_DIR="mlproject/configs/experiments"
GENERATED_DIR="mlproject/configs/generated"
LOG_DIR="tmplogs"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   VERIFYING CONFIG GENERATION & EXECUTION${NC}"
echo -e "${BLUE}   (Experiments: etth3, etth1, tabular)${NC}"
echo -e "${BLUE}   (Scope: Train -> Generate -> Eval -> Tune -> Serve_DAG)${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Iterate over all yaml files in the pipeline directory
for PIPE_PATH in "$PIPELINE_DIR"/*.yaml; do
    PIPE_FILENAME=$(basename "$PIPE_PATH")
    PIPE_NAME="${PIPE_FILENAME%.*}"

    echo -e "\n${YELLOW}-------------------------------------------------------${NC}"
    echo -e "${YELLOW}>>> Processing Pipeline: $PIPE_FILENAME${NC}"
    echo -e "${YELLOW}-------------------------------------------------------${NC}"

    # Determine Experiments to Run
    EXPERIMENTS=()
    TYPE=""
    DATA_INPUT=""
    if [[ "$PIPE_FILENAME" == *"tabular"* ]]; then
        EXPERIMENTS+=("tabular.yaml")
        TYPE="tabular"
        DATA_INPUT="mlproject/data/titanic.csv"
        echo "Detected Type: TABULAR"
    elif [[ "$PIPE_FILENAME" == *"feast"* ]]; then
        EXPERIMENTS+=("etth1_feast.yaml")
        TYPE="feast"
        DATA_INPUT="mlproject/data/ETTh1.csv"
        echo "Detected Type: FEAST"
    else
        EXPERIMENTS+=("etth3.yaml" "etth1.yaml")
        TYPE="timeseries"
        DATA_INPUT="mlproject/data/ETTh1.csv"
        echo "Detected Type: TIMESERIES (Running against: ${EXPERIMENTS[*]})"
    fi

    for EXP_NAME in "${EXPERIMENTS[@]}"; do
        EXP_CONFIG="$EXP_DIR/$EXP_NAME"
        EXP_ID="${EXP_NAME%.*}" # e.g. etth3

        echo -e "\n${BLUE}>>> Experiment: $EXP_ID${NC}"

        # 1. Train
        echo -e "${BLUE}--- [1/5] Training ($EXP_ID) ---${NC}"
        LOG_PREFIX="$LOG_DIR/${PIPE_NAME}_${EXP_ID}"

        if python -m mlproject.src.pipeline.dag_run train -e "$EXP_CONFIG" -p "$PIPE_PATH" > "${LOG_PREFIX}_train.log" 2>&1; then
            echo -e "${GREEN}Training PASS${NC}"
        else
            echo -e "${RED}Training FAILED${NC}"
            tail -n 20 "${LOG_PREFIX}_train.log"
            exit 1
        fi

        # 2. Generate ALL Configs
        echo -e "${BLUE}--- [2/5] Generating All Configs ($EXP_ID) ---${NC}"
        if python -m mlproject.src.pipeline.dag_run generate -t "$PIPE_PATH" -e "$EXP_CONFIG" --type all > "${LOG_PREFIX}_gen.log" 2>&1; then
            echo -e "${GREEN}Generation PASS${NC}"
        else
            echo -e "${RED}Generation FAILED${NC}"
            tail -n 20 "${LOG_PREFIX}_gen.log"
            exit 1
        fi

        SERVE_CONFIG="$GENERATED_DIR/${PIPE_NAME}_serve.yaml"
        EVAL_CONFIG="$GENERATED_DIR/${PIPE_NAME}_eval.yaml"
        TUNE_CONFIG="$GENERATED_DIR/${PIPE_NAME}_tune.yaml"

        # 3. Verify Eval Execution
        echo -e "${BLUE}--- [3/5] Verifying Eval Execution ($EXP_ID) ---${NC}"
        if [ ! -f "$EVAL_CONFIG" ]; then echo -e "${RED}Eval Config MISSING${NC}"; exit 1; fi

        if python -m mlproject.src.pipeline.dag_run eval -e "$EXP_CONFIG" -p "$EVAL_CONFIG" > "${LOG_PREFIX}_eval.log" 2>&1; then
            echo -e "${GREEN}Eval Execution PASS${NC}"
        else
            echo -e "${RED}Eval Execution FAILED${NC}"
            tail -n 20 "${LOG_PREFIX}_eval.log"
            exit 1
        fi

        # 4. Verify Tune Execution
        echo -e "${BLUE}--- [4/5] Verifying Tune Execution (1 trial) ($EXP_ID) ---${NC}"
        if [ ! -f "$TUNE_CONFIG" ]; then echo -e "${RED}Tune Config MISSING${NC}"; exit 1; fi

        if python -m mlproject.src.pipeline.dag_run tune -e "$EXP_CONFIG" -p "$TUNE_CONFIG" --trials 1 > "${LOG_PREFIX}_tune.log" 2>&1; then
            echo -e "${GREEN}Tune Execution PASS${NC}"
        else
            echo -e "${RED}Tune Execution FAILED${NC}"
            tail -n 20 "${LOG_PREFIX}_tune.log"
            exit 1
        fi

        # 5. Verify Serve Execution (DAG Run)
        echo -e "${BLUE}--- [5/5] Verifying Serve (DAG Run) ($EXP_ID) ---${NC}"
        if [ ! -f "$SERVE_CONFIG" ]; then echo -e "${RED}Serve Config MISSING${NC}"; exit 1; fi

        # Run dag_run serve with explict input CSV
        if python -m mlproject.src.pipeline.dag_run serve -e "$EXP_CONFIG" -p "$SERVE_CONFIG" -i "$DATA_INPUT" > "${LOG_PREFIX}_serve_dag.log" 2>&1; then
            echo -e "${GREEN}Serve (DAG) Execution PASS${NC}"
        else
            echo -e "${RED}Serve (DAG) Execution FAILED${NC}"
            tail -n 20 "${LOG_PREFIX}_serve_dag.log"
            exit 1
        fi

    done # End Experiment Loop

    echo -e "${GREEN}>>> $PIPE_FILENAME ALL CHECKS PASSED${NC}"

done

echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}   VERIFICATION COMPLETE${NC}"
echo -e "${GREEN}=======================================================${NC}"
