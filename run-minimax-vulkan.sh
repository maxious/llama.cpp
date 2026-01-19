#!/bin/bash

# Initialize Vulkan SDK
source ~/vulkan/1.4.335.0/setup-env.sh

# Run MiniMax-M2.1-MXFP4-MOE (~120GB, split across 7 GGUF files)
# Uses --fit on to automatically adjust layers to fit in VRAM
./build/bin/llama-server --model "/media/maxious/01D2E438D1AA0DD0/MiniMax-M2.1-MXFP4_MOE-GGUF/MiniMax-M2.1-MXFP4_MOE-00001-of-00007.gguf" \
  --port 5001 --host 0.0.0.0 --jinja \
  --n-gpu-layers 99 \
  --fit on \
  --device Vulkan0,Vulkan1,Vulkan2
