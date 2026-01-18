#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

# SYCL Build Script for llama.cpp with Flash Attention enhancements
# Usage: ./build-sycl.sh [--clean] [--f16] [--bf16]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAN=false
ENABLE_F16=false
ENABLE_BF16=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --f16)
            ENABLE_F16=true
            shift
            ;;
        --bf16)
            ENABLE_BF16=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--f16] [--bf16]"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [[ "$CLEAN" == true ]]; then
    rm -rf build-sycl
fi

# Create build directory
mkdir -p build-sycl
cd build-sycl

# Source oneAPI environment (required for SYCL and oneDNN)
echo "Loading oneAPI environment..."
source /opt/intel/oneapi/setvars.sh

# Configure with SYCL backend
echo "Configuring CMake..."
CMAKE_OPTS=(
    -DGGML_SYCL=ON
    -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icx
    -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icpx
    -DLLAMA_CURL=OFF
    -DGGML_SYCL_DNN=OFF
    -DMKL_SYCL_THREADING=intel_thread
)

if [[ "$ENABLE_F16" == true ]]; then
    CMAKE_OPTS+=(-DGGML_SYCL_F16=ON)
    echo "Enabling FP16 support..."
fi

if [[ "$ENABLE_BF16" == true ]]; then
    CMAKE_OPTS+=(-DGGML_SYCL_BF16=ON)
    echo "Enabling BF16 support..."
fi

cmake .. "${CMAKE_OPTS[@]}"

# Build with parallel jobs
echo "Building (this may take a while)..."
cmake --build . --config Release -j $(nproc)

echo ""
echo "Build complete! Binaries are in: build-sycl/bin/"
echo ""
echo "To list SYCL devices:"
echo "  ./build-sycl/bin/llama-ls-sycl-device"
echo ""
echo "To run a model:"
echo "  ./build-sycl/bin/llama-cli -m model.gguf -p 'Hello'"
echo ""
echo "Options used:"
echo "  --f16: Enable FP16 support"
echo "  --bf16: Enable BF16 support"
