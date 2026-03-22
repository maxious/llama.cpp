#!/bin/bash
set -euo pipefail

# Build script for SYCL backend on Intel Arc B60
# Usage: ./build-sycl.sh

# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

# Set library paths for runtime
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2025.3/lib:/opt/intel/oneapi/tbb/2025.3/lib/intel64/gcc4.8:/opt/intel/oneapi/dnnl/2025.3/lib:$LD_LIBRARY_PATH"

# Create build directory
mkdir -p build-sycl-new
cd build-sycl-new

# Configure CMake with SYCL support
cmake .. \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_DEVICE_ARCH=intel_gpu_pvc \
    -DCMAKE_C_COMPILER=/opt/intel/oneapi/2025.3/bin/icx \
    -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/2025.3/bin/icpx \
    -DOpenCL_LIBRARY=/opt/intel/oneapi/2025.3/lib/libOpenCL.so \
    -DOpenCL_INCLUDE_DIR=/opt/intel/oneapi/compiler/2025.3/include \
    -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/dnnl/2025.3" \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_SYCL_FAT=OFF

# Build llama-bench
cmake --build . --target llama-bench -j$(nproc)

echo "Build complete: build-sycl-new/bin/llama-bench"
