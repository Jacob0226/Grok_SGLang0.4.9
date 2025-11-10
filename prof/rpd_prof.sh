#!/bin/bash
set -x

export BS=512
export ILEN=256
export OLEN=2048
export PageSize=16

MODEL_PATH="/data/huggingface/hub/amd/grok-1-W4A8KV8"
TOKENIZER_PATH="/data/huggingface/hub/Xenova/grok-1-tokenizer"

SGLANG_ARGS="--batch-size ${BS} --input ${ILEN} --output ${OLEN} --tp 8 --page-size ${PageSize} \
             --quantization fp8 --trust-remote-code \
             --model ${MODEL_PATH} \
             --tokenizer-path ${TOKENIZER_PATH} \
             --attention-backend aiter --enable-decode-prof"


run_benchmark() {
    local version=$1
    echo "======================================================"
    echo "Starting Benchmark for QKV Version: ${version}"
    echo "======================================================"

    export QKV_VERSION="${version}"
    export OUT="E2E_bs${BS}_ilen${ILEN}_olen${OLEN}_PageSize${PageSize}_${QKV_VERSION}"
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 python -m \
        sglang.bench_one_batch ${SGLANG_ARGS} 2>&1 | tee "${OUT}.log"
    
    mv trace.rpd ${OUT}.rpd
    sqlite3 ${OUT}.rpd ".mode csv" ".header on" ".output ${OUT}.csv" "select * from top;" ".output stdout" # Convert trace.rpd into csv
    python3 /app/rocmProfileData/tools/rpd2tracing.py  ${OUT}.rpd  ${OUT}.json
    python ~/Grok_SGLang0.4.9/prof/rpd_trace_helper.py -i ${OUT}.json
}

run_benchmark "GOLDEN"
run_benchmark "Jacob"