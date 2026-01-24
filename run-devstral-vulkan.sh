#!/bin/bash

# Initialize Vulkan SDK
source /opt/vulkan/1.4.335.0/setup-env.sh

# Run Devstral 24B with Vulkan backend
# Exclude iGPU (Vulkan3), use discrete GPUs: Intel Arc B60 (Vulkan0,1) + NVIDIA RTX 5080 (Vulkan2)
./build/bin/llama-server --model ./koboldcpp/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf \
  --port 5000 --host 0.0.0.0 --jinja \
  --threads -1 \
  --ctx-size 2048 \
  --n-gpu-layers 99 \
  --cache-ram -1 \
  --device Vulkan0,Vulkan1,Vulkan2 \
  --temp 0.15
