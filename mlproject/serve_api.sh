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

# Parse arguments
SERVE_CONFIG=""

show_help() {
    echo "Usage: ./serve_api.sh [OPTIONS] <serve_config.yaml>"
    echo ""
    echo "Options:"
    echo "  -f, --framework FRAMEWORK   Framework: fastapi or ray (default: fastapi)"
    echo "  -p, --port PORT            Port number (default: 8000)"
    echo "  -h, --host HOST            Host address (default: 0.0.0.0)"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  # FastAPI on port 8000"
    echo "  ./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml"
    echo ""
    echo "  # Ray Serve on port 9000"
    echo "  ./serve_api.sh -f ray -p 9000 mlproject/configs/generated/standard_train_serve.yaml"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --help)
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

# Check if file exists
if [ ! -f "$SERVE_CONFIG" ]; then
    echo -e "${YELLOW}Error: File not found: $SERVE_CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Serve config: $SERVE_CONFIG"
echo "  Framework: $FRAMEWORK"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

# Run the Python command
echo -e "${BLUE}Starting server...${NC}"
echo ""

python -m mlproject.serve.run_generated_api \
    --serve-config "$SERVE_CONFIG" \
    --framework "$FRAMEWORK" \
    --host "$HOST" \
    --port "$PORT"
