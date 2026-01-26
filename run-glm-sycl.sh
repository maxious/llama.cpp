#!/bin/bash

# GLM-4.7-Flash SYCL Run Script
# Memory Requirements: ~18GB for 4-bit quantization, 24GB RAM/VRAM/unified memory (32GB for full precision)
# Maximum Context Window: 202,752 tokens
# Note: llama.cpp fixed a looping bug - please re-download the model for better outputs
# Model: unsloth/GLM-4.7-Flash-GGUF (e.g., UD-Q4_K_XL or UD-Q3_K_XL)

# Initialize Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh intel64

# Enable support for >4GB allocations on Intel GPUs
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# Enable SYSMAN for better memory management
export ZES_ENABLE_SYSMAN=1
# Disable shared USM - causes incorrect output with multi-GPU
export GGML_SYCL_SHARED_USM=0

# Check for required GPU group permissions
if ! groups | grep -qwE "(render|video)"; then
    echo "WARNING: User is not in 'render' or 'video' groups."
    echo "This may cause 'UR_RESULT_ERROR_UNINITIALIZED' errors."
    echo "Run: sudo usermod -aG render \$USER && sudo usermod -aG video \$USER"
    echo "Then log out and back in."
fi

# Debug logging
#export GGML_SYCL_DEBUG=1

# SINGLE GPU MODE OPTION
# Uncomment the line below to force usage of only the first GPU (Device 0)
# This helps rule out multi-GPU synchronization/P2P issues
#export ONEAPI_DEVICE_SELECTOR=level_zero:0

./build-sycl/bin/llama-server \
    --model koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf \
    --port 5000 --host 0.0.0.0 --jinja \
    --threads -1 \
    --cache-ram -1 \
    --fit on \
     --temp 0.7 --top-p 1.0 --min-p 0.01
