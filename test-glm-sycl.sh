#!/bin/bash

# GLM-4.7-Flash SYCL Test Script - single query for debugging

source /opt/intel/oneapi/setvars.sh intel64

export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
#export ONEAPI_DEVICE_SELECTOR=level_zero:0
./build-sycl/bin/llama-completion \
    --model ~/koboldcpp/GLM-4.7-Flash-REAP-23B-A3B-Q8_0.gguf --mmap \
    --prompt "What is the capital of france?" --n-predict 8  --no-conversation \
    --ctx-size 512 --flash-attn on \
    --batch-size 64 --ubatch-size 64 \
    --temp 0.7 --top-p 1.0 --min-p 0.01
