#!/bin/bash

# Initialize Vulkan SDK
source ~/vulkan/1.4.335.0/setup-env.sh

# Run Qwen 0.6B with Vulkan backend
# Exclude iGPU (Vulkan3), use discrete GPUs
./build/bin/llama-server --model ./koboldcpp/Qwen3-0.6B-ICM-DPO.f16.gguf \
  --port 5000 --host 0.0.0.0 --jinja \
  --n-gpu-layers 99 \
  -c 2048 \
  --device Vulkan0,Vulkan1,Vulkan2
