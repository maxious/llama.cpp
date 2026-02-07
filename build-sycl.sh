#!/usr/bin/env bash
#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT
#
# SYCL Build Script for llama.cpp
# Usage: ./build-sycl.sh [--clean] [--no-f16] [--no-bf16] [--asan] [--ubsan] [--sanitize] [--aot <target>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAN=false
ENABLE_F16=true
ENABLE_BF16=true
ENABLE_ASAN=false
ENABLE_UBSAN=false
ENABLE_AOT=""
BUILD_TYPE="Release"

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
        --no-f16)
            ENABLE_F16=false
            shift
            ;;
        --bf16)
            ENABLE_BF16=true
            shift
            ;;
        --no-bf16)
            ENABLE_BF16=false
            shift
            ;;
        --asan)
            ENABLE_ASAN=true
            BUILD_TYPE="RelWithDebInfo"
            shift
            ;;
        --ubsan)
            ENABLE_UBSAN=true
            BUILD_TYPE="RelWithDebInfo"
            shift
            ;;
        --sanitize)
            ENABLE_ASAN=true
            ENABLE_UBSAN=true
            BUILD_TYPE="RelWithDebInfo"
            shift
            ;;
        --aot)
            shift
            if [[ -z "$1" || "$1" == --* ]]; then
                # Default to BMG (Battlemage/Xe2) if no target specified
                ENABLE_AOT="intel_gpu_bmg_g21"
            else
                ENABLE_AOT="$1"
                shift
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--no-f16] [--no-bf16] [--asan] [--ubsan] [--sanitize] [--aot <target>]"
            echo ""
            echo "Note: FP16 and BF16 are enabled by default for Intel GPUs."
            echo ""
            echo "Sanitizer options:"
            echo "  --asan     Enable Address Sanitizer (ASAN) for memory error detection"
            echo "  --ubsan    Enable Undefined Behavior Sanitizer (UBSAN)"
            echo "  --sanitize Enable both ASAN and UBSAN (equivalent to --asan --ubsan)"
            echo ""
            echo "AOT (Ahead-of-Time) compilation:"
            echo "  --aot [target]  Enable AOT compilation for specified Intel GPU target"
            echo "                  Default: intel_gpu_bmg_g21 (Battlemage/Xe2)"
            echo "                  Other options: intel_gpu_pvc, intel_gpu_acm_g10, etc."
            echo "                  See: icpx --help for -fsycl-targets options"
            echo ""
            echo "Example with AOT for Battlemage:"
            echo "  ./build-sycl.sh --aot"
            echo ""
            echo "Example with sanitizers for debugging:"
            echo "  ./build-sycl.sh --asan"
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

# Clean stale library symlinks that can cause "failed to create symbolic link" errors
# These may block linking if they exist from a previous build
if [ -d "build-sycl/bin" ]; then
    rm -f build-sycl/bin/libggml.so* build-sycl/bin/libggml-sycl.so* 2>/dev/null || true
fi

cd build-sycl

# Source oneAPI environment (required for SYCL and oneDNN)
echo "Loading oneAPI environment..."
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1 || true

# Configure with SYCL backend
echo "Configuring CMake..."
CMAKE_OPTS=(
    -DGGML_SYCL=ON
    -DGGML_SYCL_GRAPH=OFF
    -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icx
    -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icpx
    -DLLAMA_CURL=OFF
    -DGGML_SYCL_DNN=ON
    -DMKL_SYCL_THREADING=intel_thread
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
)

if [[ "$ENABLE_F16" == true ]]; then
    CMAKE_OPTS+=(-DGGML_SYCL_F16=ON)
    echo "Enabling FP16 support..."
fi

if [[ "$ENABLE_BF16" == true ]]; then
    CMAKE_OPTS+=(-DGGML_SYCL_BF16=ON)
    echo "Enabling BF16 support..."
fi

if [[ "$ENABLE_ASAN" == true ]]; then
        SANITIZER_FLAGS="${SANITIZER_FLAGS} -fsanitize=address -fno-omit-frame-pointer"
        CMAKE_OPTS+=(-DCMAKE_CXX_FLAGS="-fsycl -fsanitize=address -fno-omit-frame-pointer -g")
        CMAKE_OPTS+=(-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address")
        CMAKE_OPTS+=(-DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g")
        echo "Enabling Address Sanitizer (ASAN)..."
        echo ""
        echo "IMPORTANT: For GPU memory issues, run with device-side ASAN:"
        echo "  export UR_LAYER_ASAN_OPTIONS=\"quarantine_size_mb:16;redzone:64\""
        echo "  export CLANG_TOOLCHAIN_PROGRAM_TIMEOUT=600"
        echo "  ./build-sycl/bin/llama-cli -m model.gguf -p 'test'"
        echo ""
        echo "Device-side ASAN makes GPU execution sequential and may reduce workgroup size."
    fi

if [[ "$ENABLE_UBSAN" == true ]]; then
    SANITIZER_FLAGS="${SANITIZER_FLAGS} -fsanitize=undefined"
    if [[ "$ENABLE_ASAN" != true ]]; then
        # Only add to CXX_FLAGS if ASAN wasn't already set
        CMAKE_OPTS+=(-DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer -g")
        CMAKE_OPTS+=(-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined")
    fi
    echo "Enabling Undefined Behavior Sanitizer (UBSAN)..."
fi

if [[ -n "$ENABLE_AOT" ]]; then
    echo "Enabling AOT compilation for target: $ENABLE_AOT"
    CMAKE_OPTS+=(-DGGML_SYCL_DEVICE_ARCH="$ENABLE_AOT")
fi

cmake .. "${CMAKE_OPTS[@]}"

# Build with parallel jobs
echo "Building (this may take a while)..."
cmake --build . --config ${BUILD_TYPE} -j $(nproc)

echo ""
echo "Build complete! Binaries are in: build-sycl/bin/"
echo ""
echo "To list SYCL devices:"
echo "  ./build-sycl/bin/llama-ls-sycl-device"
echo ""
echo "To run a model (with required environment setup):"
echo ""
echo "  source /opt/intel/oneapi/setvars.sh intel64"
echo "  export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1"
echo "  export ZES_ENABLE_SYSMAN=1"
echo "  ./build-sycl/bin/llama-completion \\"
echo "      --model model.gguf --mmap \\"
echo "      --prompt 'What is the capital of france?' --n-predict 32 --no-conversation \\"
echo "      --ctx-size 512 --flash-attn on \\"
echo "      --batch-size 64 --ubatch-size 64 \\"
echo "      --temp 0.7 --top-p 1.0 --min-p 0.01"
echo ""
echo "Options used:"
echo "  --f16: Enable FP16 support"
echo "  --bf16: Enable BF16 support"
if [[ "$ENABLE_ASAN" == true ]]; then
    echo "  --asan: Address Sanitizer enabled"
fi
if [[ "$ENABLE_UBSAN" == true ]]; then
    echo "  --ubsan: Undefined Behavior Sanitizer enabled"
fi
