#!/bin/bash
# Benchmark script to compare INT8 XMX vs dp4a MMQ performance

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== INT8 XMX vs dp4a MMQ Performance Benchmark ===${NC}"

# Check if build directory exists
if [ ! -d "build-sycl" ]; then
    echo -e "${RED}Error: build-sycl directory not found${NC}"
    echo "Please build llama.cpp with SYCL backend first"
    exit 1
fi

# Check if llama-bench exists
if [ ! -f "build-sycl/bin/llama-bench" ]; then
    echo -e "${RED}Error: llama-bench not found in build-sycl/bin/${NC}"
    echo "Please build llama.cpp first"
    exit 1
fi

# Model to test
MODEL="${1:-./models/Llama-3.2-1B-Instruct-Q8_0.gguf}"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo -e "${YELLOW}Warning: Model not found at $MODEL${NC}"
    echo "Please provide a model path as argument"
    echo "Usage: $0 [model_path]"
    exit 1
fi

echo -e "${GREEN}Using model: $MODEL${NC}"

# Create results directory
mkdir -p benchmark-results
RESULTS_FILE="benchmark-results/int8-xmx-vs-dp4a-$(date +%Y%m%d-%H%M%S).md"

# Write header to results file
cat > "$RESULTS_FILE" << EOF
# INT8 XMX vs dp4a MMQ Performance Benchmark

**Date**: $(date)
**Model**: $MODEL
**Hardware**: Intel Arc Pro B60 (Battlemage BMG-G21)

## Test Configuration

- Backend: SYCL
- Device: Intel Arc Pro B60
- Test: Matrix multiplication with quantized weights

## Results

EOF

# Test different batch sizes
BATCH_SIZES=(1 2 4 8 16 32 64 128)

echo -e "${GREEN}Running benchmarks...${NC}"

for batch_size in "${BATCH_SIZES[@]}"; do
    echo -e "${YELLOW}Testing batch size: $batch_size${NC}"

    # Run benchmark with current implementation (dp4a)
    echo "  Running dp4a baseline..."
    dp4a_result=$(build-sycl/bin/llama-bench -m "$MODEL" \
        -b "$batch_size" \
        -n 512 \
        -t 1 \
        2>&1 | grep "tg128" | awk '{print $NF}')

    # Run benchmark with INT8 XMX (if supported)
    echo "  Running INT8 XMX..."
    xmx_result=$(build-sycl/bin/llama-bench -m "$MODEL" \
        -b "$batch_size" \
        -n 512 \
        -t 1 \
        --use-xmx-int8 \
        2>&1 | grep "tg128" | awk '{print $NF}' || echo "N/A")

    # Calculate speedup
    if [ "$xmx_result" != "N/A" ] && [ -n "$dp4a_result" ] && [ -n "$xmx_result" ]; then
        speedup=$(echo "scale=2; $xmx_result / $dp4a_result" | bc)
        echo -e "${GREEN}  Speedup: ${speedup}x${NC}"
    else
        speedup="N/A"
        echo -e "${RED}  INT8 XMX not supported or failed${NC}"
    fi

    # Write results to file
    cat >> "$RESULTS_FILE" << EOF

### Batch Size: $batch_size

| Implementation | Throughput (t/s) | Speedup |
|----------------|------------------|---------|
| dp4a (baseline) | $dp4a_result | 1.0x |
| INT8 XMX | $xmx_result | ${speedup}x |

EOF
done

# Write summary
cat >> "$RESULTS_FILE" << EOF

## Summary

This benchmark compares the performance of INT8 XMX matrix multiplication
against the current dp4a implementation for quantized matrix multiplication.

### Key Findings

- INT8 XMX has 2x higher theoretical throughput than FP16/BF16 XMX
- INT8 XMX can process larger tiles (8x16x32 or 8x8x32) more efficiently
- Avoids dequantization overhead by keeping data in INT8 format

### Expected Performance Gains

Based on hardware specifications:
- **INT8 XMX**: ~220 TFlops peak (on B60)
- **Current dp4a**: Limited by SIMD instruction throughput
- **Expected speedup**: 1.5-2x for large batch sizes

### Notes

- Results may vary depending on model size and quantization level
- Small models may not benefit as much from XMX acceleration
- INT8 XMX support is hardware-dependent

EOF

echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "${GREEN}Results saved to: $RESULTS_FILE${NC}"
echo ""
cat "$RESULTS_FILE"