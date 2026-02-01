#!/bin/bash

#     --model koboldcpp/GLM-4.7-Flash-MXFP4_MOE.gguf \
 ./build-cuda/bin/llama-server \
     --model ../koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf \
     --port 5000 --host 0.0.0.0 --jinja \
     --fit on \
     -ctk q4_0 -ctv q4_0 \
     --temp 0.7 --top-p 1.0 --min-p 0.01
