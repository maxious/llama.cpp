#!/bin/bash

# GLM-4.7-Flash SYCL Test Script - single query for debugging

source /opt/intel/oneapi/setvars.sh intel64

export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

./build-sycl/bin/llama-cli \
    --model koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf \
    --prompt "Hello, how are you?" \
    --n-predict 32 \
    --fit on \
    --temp 0.7 --top-p 1.0 --min-p 0.01
