#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
    echo "Loaded environment variables from $ENV_FILE"
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Run vLLM server
echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"

vllm serve "$MODEL_NAME" --port "$PORT" --max_model_len "$MAX_MODEL_LEN"
