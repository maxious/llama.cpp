#!/bin/bash

# Initialize Vulkan SDK
source ~/vulkan/1.4.335.0/setup-env.sh

# Run MiniMax-M2.1-REAP-50.i1-IQ3_XXS.gguf with conservative GPU settings
# Reduce layers and context to fit in VRAM
./build/bin/llama-server --model ./koboldcpp/MiniMax-M2.1-REAP-50.i1-IQ3_XXS.gguf \
  --port 5001 --host 0.0.0.0 --jinja \
  --n-gpu-layers 32 \
  --ctx-size 16384 \
  --fit on
