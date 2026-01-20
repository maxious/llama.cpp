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

# Run llama-server (SYCL version)
./build/bin/llama-server --model koboldcpp/GLM-4.7-Flash-Q4_K_M.gguf \
  --port 5000 --host 0.0.0.0 --jinja \
  --threads -1 \
  --n-gpu-layers 99 \
  --cache-ram -1 \
  -sm layer \
  --seed 3407 \
  --prio 2 \
  --temp 0.15
