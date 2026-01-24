#!/bin/bash

# Initialize Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh intel64

# Enable support for >4GB allocations on Intel GPUs
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# Enable SYSMAN for better memory management
export ZES_ENABLE_SYSMAN=1
# Run MiniMax-M2.1-MXFP4-MOE (~120GB, split across 7 GGUF files)
# Uses --fit on to automatically adjust layers to fit in VRAM
./build-sycl/bin/llama-server --model ./koboldcpp/MiniMax-M2.1-REAP-50.i1-IQ3_XXS.gguf \
 --port 5000 --host 0.0.0.0 --jinja \
  --n-gpu-layers 99 \
  --fit on
