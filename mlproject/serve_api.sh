#!/bin/bash
# Quick launcher for serve APIs

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Serve API Quick Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default values
FRAMEWORK="fastapi"
PORT=8000
HOST="0.0.0.0"
EXPERIMENT_CONFIG=""
ALIAS="production"

# Parse arguments
SERVE_CONFIG=""

show_help() {
    echo "Usage: ./serve_api.sh [OPTIONS] <serve_config.yaml>"
    echo ""
    echo "Options:"
    echo "  -e, --experiment EXPERIMENT Experiment config (e.g., etth1.yaml)"
    echo "  -f, --framework FRAMEWORK   Framework: fastapi or ray (default: fastapi)"
    echo "  -p, --port PORT            Port number (default: 8000)"
    echo "  -h, --host HOST            Host address (default: 0.0.0.0)"
    echo "  -a, --alias ALIAS          MLflow alias (default: production)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  # FastAPI on port 8000 with latest alias"
    echo "  ./serve_api.sh -e mlproject/configs/experiments/etth1.yaml -a latest mlproject/configs/generated/standard_train_serve.yaml"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT_CONFIG="$2"
            shift 2
            ;;
        -f|--framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -a|--alias)
            ALIAS="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            ;;
        *)
            SERVE_CONFIG="$1"
            shift
            ;;
    esac
done

# Check if serve config provided
if [ -z "$SERVE_CONFIG" ]; then
    echo -e "${YELLOW}Error: Serve config not provided${NC}"
    echo ""
    show_help
fi

# Check if experiment config provided
if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo -e "${YELLOW}Error: Experiment config not provided${NC}"
    echo -e "${YELLOW}Use -e to specify experiment config (e.g., -e mlproject/configs/experiments/etth1.yaml)${NC}"
    echo ""
    show_help
fi

# Check if files exist
if [ ! -f "$SERVE_CONFIG" ]; then
    echo -e "${YELLOW}Error: Serve config not found: $SERVE_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo -e "${YELLOW}Error: Experiment config not found: $EXPERIMENT_CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Experiment config: $EXPERIMENT_CONFIG"
echo "  Serve config: $SERVE_CONFIG"
echo "  Framework: $FRAMEWORK"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Alias: $ALIAS"
echo ""

# Run the Python command
echo -e "${BLUE}Starting server...${NC}"
echo ""

python -m mlproject.serve.run_generated_api \
    --serve-config "$SERVE_CONFIG" \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --framework "$FRAMEWORK" \
    --host "$HOST" \
    --port "$PORT" \
    --alias "$ALIAS"
