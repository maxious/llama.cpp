#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

# Enable oneAPI environment (required for SYCL and oneDNN)
source /opt/intel/oneapi/setvars.sh

# Clean and create build directory
rm -rf build
mkdir -p build
cd build

# Configure with SYCL backend for Intel GPU
# - Uses icx/icpx Intel compilers
# - Enables Flash Attention (fattn) support for faster attention computation
# - Disables CURL for offline builds (optional)

#for FP16
#cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_CURL=OFF # faster for long-prompt inference

#for FP32
cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_CURL=OFF

# Build all binaries with parallel jobs
cmake --build . --config Release -j $(nproc)
