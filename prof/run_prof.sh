#!/bin/bash
set -x

BS=(8 16 32 64 128 256 512)
out_folder=bs512
mkdir -p $out_folder

# Iterate from 1 to 8 to run the benchmark for each kernel version
for bs in "${BS[@]}"; do
    # Run the second benchmark with profiling enabled
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 \
    python -m sglang.bench_one_batch --batch-size $bs --input 256 --output 2048 --tp 8 \
        --quantization fp8 --trust-remote-code --attention-backend aiter \
        --model /data/grok-1-W4A8KV8  \
        --tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer \
        --enable-decode-prof

    sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv"   "select * from top;" ".output stdout"
    mv trace.rpd $out_folder/trace_bs${bs}_i256_o2048.rpd
    mv trace.csv $out_folder/trace_bs${bs}_i256_o2048.csv
done