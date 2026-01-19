#!/bin/bash

# Initialize Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh intel64

# Enable support for >4GB allocations on Intel GPUs
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# Enable SYSMAN for better memory management
export ZES_ENABLE_SYSMAN=1

# Check for required GPU group permissions
if ! groups | grep -qwE "(render|video)"; then
    echo "WARNING: User is not in 'render' or 'video' groups."
    echo "This may cause 'UR_RESULT_ERROR_UNINITIALIZED' errors."
    echo "Run: sudo usermod -aG render \$USER && sudo usermod -aG video \$USER"
    echo "Then log out and back in."
fi

# Debug logging
export GGML_SYCL_DEBUG=1

# SINGLE GPU MODE OPTION
# Uncomment the line below to force usage of only the first GPU (Device 0)
# This helps rule out multi-GPU synchronization/P2P issues
# export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Run llama-server (SYCL version) with safe parameters for 24B model
# -ngl 7: Limited to 7 GPU layers (8+ causes UR_RESULT_ERROR_DEVICE_LOST)
# -c 2048: Smaller context to fit within GPU memory limits
./build/bin/llama-server --model ./koboldcpp/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf \
  --port 5000 --host 0.0.0.0 --jinja \
  --threads -1 \
  --ctx-size 2048 \
  --n-gpu-layers 7 \
  --cache-ram -1 \
  -sm layer \
  --seed 3407 \
  --prio 2 \
  --temp 0.15