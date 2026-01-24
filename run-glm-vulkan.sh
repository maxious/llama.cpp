#!/bin/bash

# GLM-4.7-Flash Vulkan Run Script
# Memory Requirements: ~18GB for 4-bit quantization, 24GB RAM/VRAM/unified memory (32GB for full precision)
# Maximum Context Window: 202,752 tokens
# Note: llama.cpp fixed a looping bug - please re-download the model for better outputs
# Model: unsloth/GLM-4.7-Flash-GGUF (e.g., UD-Q4_K_XL or UD-Q3_K_XL)

# Initialize Vulkan SDK
source /opt/vulkan/1.4.335.0/setup-env.sh

# Exclude iGPU (Vulkan3), use discrete GPUs: Intel Arc B60 (Vulkan1,2) + NVIDIA RTX 5080 (Vulkan0)

#     --model koboldcpp/GLM-4.7-Flash-MXFP4_MOE.gguf \
     #--device Vulkan0,Vulkan1,Vulkan2 \
 ./build/bin/llama-server \
     --model koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf \
     --port 5000 --host 0.0.0.0 --jinja \
     --threads -1 \
     --cache-ram -1 \
     --fit on \
     --device Vulkan1,Vulkan2 \
     --temp 0.7 --top-p 1.0 --min-p 0.01
