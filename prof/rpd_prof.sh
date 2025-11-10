#!/bin/bash
set -x

# --- Fixed Environment Variables ---
export BS=512     # Batch Size
export ILEN=256   # Input Length (Context Length)
export OLEN=2048  # Output Length (Max generated tokens)

# Define the list of PageSize values to loop through
PAGE_SIZES=(1 16)

MODEL_PATH="/data/huggingface/hub/amd/grok-1-W4A8KV8"
TOKENIZER_PATH="/data/huggingface/hub/Xenova/grok-1-tokenizer"

# Function to run the benchmark for a specific version and page size
# Arguments: $1 = QKV_VERSION (GOLDEN or Jacob), $2 = PageSize
run_benchmark() {
    local version=$1       # QKV_VERSION (e.g., GOLDEN or Jacob)
    local page_size=$2     # Current PageSize value

    # Construct SGLANG_ARGS based on the current page_size
    SGLANG_ARGS="--batch-size ${BS} --input ${ILEN} --output ${OLEN} --tp 8 --page-size ${page_size} \
                 --quantization fp8 --trust-remote-code \
                 --model ${MODEL_PATH} \
                 --tokenizer-path ${TOKENIZER_PATH} \
                 --attention-backend aiter --enable-decode-prof"
                 
    echo "======================================================"
    echo "Starting Benchmark for Version: ${version} | PageSize: ${page_size}"
    echo "======================================================"

    export QKV_VERSION="${version}"
    export PageSize="${page_size}" # Export PageSize for file naming
    
    # Create the output filename (now includes PageSize for uniqueness)
    export OUT="E2E_bs${BS}_ilen${ILEN}_olen${OLEN}_PageSize${PageSize}_${QKV_VERSION}"
    
    # Execute the benchmark run
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 python -m \
        sglang.bench_one_batch ${SGLANG_ARGS} 2>&1 | tee "${OUT}.log"
    
    # --- RPD Trace File Processing ---
    # 1. Rename the RPD file
    mv trace.rpd ${OUT}.rpd
    # 2. Convert trace.rpd into a CSV file (top-level view)
    sqlite3 ${OUT}.rpd ".mode csv" ".header on" ".output ${OUT}.csv" "select * from top;" ".output stdout"
    # 3. Convert trace.rpd to JSON tracing format
    python3 /app/rocmProfileData/tools/rpd2tracing.py ${OUT}.rpd ${OUT}.json 2>&1 | tee -a "${OUT}.log"
    # 4. Helper script to process the JSON trace
    python ~/Grok_SGLang0.4.9/prof/rpd_trace_helper.py -i ${OUT}.json 2>&1 | tee -a "${OUT}.log"
}

# --- Main Execution Loop ---
# Loop through all defined Page Sizes
for PS in "${PAGE_SIZES[@]}"; do
    run_benchmark "GOLDEN" "${PS}"
    run_benchmark "Jacob" "${PS}"
done