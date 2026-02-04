#!/bin/bash

# GLM-4.7-Flash SYCL Test Script - tests XMX on/off flash attention

source /opt/intel/oneapi/setvars.sh intel64

export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1

MODEL=models/koboldcpp/GLM-4.7-Flash-REAP-23B-A3B-Q8_0.gguf
PROMPT="[gMASK]<sop><|user|>\nWhat is the capital of France?<|assistant|>\n"

echo "=== Testing XMX Flash Attention (default) ==="
export GGML_SYCL_FLASH_ATTN_XMX=1
./build-sycl/bin/llama-completion \
    --model "$MODEL" \
    --prompt "$PROMPT" -n 64 --no-conversation \
    --ctx-size 512 --flash-attn on \
    --temp 0.7 --top-p 1.0 --min-p 0.01

echo ""
echo "=== Testing MKL Flash Attention (XMX disabled) ==="
export GGML_SYCL_FLASH_ATTN_XMX=0
./build-sycl/bin/llama-completion \
    --model "$MODEL" \
    --prompt "$PROMPT" -n 64 --no-conversation \
    --ctx-size 512 --flash-attn on \
    --temp 0.7 --top-p 1.0 --min-p 0.01
