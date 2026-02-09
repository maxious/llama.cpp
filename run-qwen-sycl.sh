#!/bin/bash

# GLM-4.7-Flash SYCL Run Script
# Memory Requirements: ~18GB for 4-bit quantization, 24GB RAM/VRAM/unified memory (32GB for full precision)
# Maximum Context Window: 202,752 tokens
# Note: llama.cpp fixed a looping bug - please re-download the model for better outputs
# Model: unsloth/GLM-4.7-Flash-GGUF (e.g., UD-Q4_K_XL or UD-Q3_K_XL)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/llama-server.pid"
LOG_FILE="$SCRIPT_DIR/llama-server.log"

# Handle stop command
if [ "$1" = "stop" ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping llama-server (PID $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            echo "Stopped."
        else
            echo "Process $PID not running, removing stale PID file."
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found."
    fi
    exit 0
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "llama-server already running (PID $PID). Use '$0 stop' to stop it."
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Initialize Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh intel64

# Enable support for >4GB allocations on Intel GPUs
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# Enable SYSMAN for better memory management
export ZES_ENABLE_SYSMAN=1
#export ONEAPI_DEVICE_SELECTOR=level_zero:0
# Check for required GPU group permissions
if ! groups | grep -qwE "(render|video)"; then
    echo "WARNING: User is not in 'render' or 'video' groups."
    echo "This may cause 'UR_RESULT_ERROR_UNINITIALIZED' errors."
    echo "Run: sudo usermod -aG render \$USER && sudo usermod -aG video \$USER"
    echo "Then log out and back in."
fi


# Debug logging
#export GGML_SYCL_DEBUG=1
#export GGML_SYCL_FLASH_ATTN_DEBUG=1
#export LLAMA_BATCH_DEBUG=1
#export LLAMA_KV_CACHE_DEBUG=1
#export LLAMA_GRAPH_INPUT_DEBUG=1
#export LLAMA_GRAPH_RESULT_DEBUG=1

echo "Starting llama-server in background..."
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"

nohup ./build-sycl/bin/llama-server \
    --model models/koboldcpp/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf \
    --port 5000 --host 0.0.0.0 \
    --fit on -c 65536 --fit-ctx 65536 --flash-attn on \
    --temp 0 \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started with PID $(cat "$PID_FILE")"
echo ""
echo "Usage:"
echo "  View logs: tail -f $LOG_FILE"
echo "  Stop server: $0 stop"
echo "  Test API: curl http://localhost:5000/v1/models"
