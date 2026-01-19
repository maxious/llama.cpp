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

# SINGLE GPU MODE - Commented out to enable DUAL GPU
# export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Run llama-server with Qwen 0.6B model (fully GPU accelerated)
# -ngl 99: All layers fit in GPU memory (1.2GB model)
# -c 2048: Context size
./build/bin/llama-server --model ./koboldcpp/Qwen3-0.6B-ICM-DPO.f16.gguf \
  --port 5000 --host 0.0.0.0 --jinja \
  --n-gpu-layers 99 \
  -c 2048 \
  -fit on \
  --verbose