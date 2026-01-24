#!/bin/bash

# GLM-4.7-Flash SYCL Test Script - single query for debugging

source /opt/intel/oneapi/setvars.sh intel64

export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

./build-sycl/bin/llama-completion \
    --model ~/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf \
    --prompt "Hello" --n-predict 1 --ctx-size 512 --flash-attn on --temp 0.7