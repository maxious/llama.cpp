#!/bin/bash

# Comprehensive Flash Attention Benchmark Script
# Tests different batch sizes and configurations with Llama and GLM models

set -e

# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh intel64 2>/dev/null

# Set environment variables for Intel Arc
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Configuration
LLAMA_BENCH="./build-sycl/bin/llama-bench"
RESULTS_FILE="flash-attn-bench-results-$(date +%Y%m%d-%H%M%S).md"
REPEATITIONS=3

# Model paths
LLAMA_MODEL="models/koboldcpp/Llama-3.2-1B.Q8_0.gguf"
GLM_MODEL="models/koboldcpp/GLM-4.7-Flash-SynthLabs-REAP-25-Q4_K_M.gguf"

# Test configurations
# Format: "prompt_len:gen_len:batch_sizes"
TEST_CONFIGS=(
    "128:32:1,8,16,32,64,128"
    "512:128:1,8,16,32,64"
    "1024:256:1,8,16,32"
)

# Function to run benchmark
run_benchmark() {
    local model=$1
    local model_name=$2
    local prompt_len=$3
    local gen_len=$4
    local batch_size=$5
    local flash_attn=$6

    echo "Running: $model_name | p=$prompt_len n=$gen_len b=$batch_size fa=$flash_attn" | tee -a "$RESULTS_FILE"

    $LLAMA_BENCH \
        -m "$model" \
        -p "$prompt_len" \
        -n "$gen_len" \
        -b "$batch_size" \
        -ub "$batch_size" \
        -fa "$flash_attn" \
        -ngl 99 \
        -r "$REPEATITIONS" \
        -o md \
        2>&1 | tee -a "$RESULTS_FILE"

    echo "" | tee -a "$RESULTS_FILE"
}

# Create results file
cat > "$RESULTS_FILE" << 'EOF'
# Flash Attention Benchmark Results

**Hardware**: Intel Arc Pro B60 (Battlemage BMG-G21)
**Date**: $(date)
**Backend**: SYCL

EOF

echo "Starting benchmarks..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Test Llama model
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Testing Llama-3.2-1B (Q8_0)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r prompt_len gen_len batch_sizes <<< "$config"
    IFS=',' read -ra bs_array <<< "$batch_sizes"

    echo "## Prompt Length: $prompt_len, Generation: $gen_len" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    # Test with flash attention ON
    echo "### Flash Attention: ON" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    for bs in "${bs_array[@]}"; do
        run_benchmark "$LLAMA_MODEL" "Llama-3.2-1B" "$prompt_len" "$gen_len" "$bs" 1
    done

    # Test with flash attention OFF
    echo "### Flash Attention: OFF" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    for bs in "${bs_array[@]}"; do
        run_benchmark "$LLAMA_MODEL" "Llama-3.2-1B" "$prompt_len" "$gen_len" "$bs" 0
    done

    echo "" | tee -a "$RESULTS_FILE"
done

# Test GLM model
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Testing GLM-4.7-Flash (Q4_K_M)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r prompt_len gen_len batch_sizes <<< "$config"
    IFS=',' read -ra bs_array <<< "$batch_sizes"

    echo "## Prompt Length: $prompt_len, Generation: $gen_len" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"

    # Test with flash attention ON
    echo "### Flash Attention: ON" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    for bs in "${bs_array[@]}"; do
        run_benchmark "$GLM_MODEL" "GLM-4.7-Flash" "$prompt_len" "$gen_len" "$bs" 1
    done

    # Test with flash attention OFF
    echo "### Flash Attention: OFF" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    for bs in "${bs_array[@]}"; do
        run_benchmark "$GLM_MODEL" "GLM-4.7-Flash" "$prompt_len" "$gen_len" "$bs" 0
    done

    echo "" | tee -a "$RESULTS_FILE"
done

echo "========================================" | tee -a "$RESULTS_FILE"
echo "Benchmark complete!" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
