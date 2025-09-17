#!/bin/bash
set -x

bs=512
out_folder=tmp #bs${bs}_CodeSeg
model_path=/data/grok-1-W4A8KV8
mkdir -p $out_folder
Code_SEGMENT=(6-1 7-1 7-2)

# Iterate from 1 to 8 to run the benchmark for each kernel version
for seg in "${Code_SEGMENT[@]}"; do
    # Remove previous build to ensure a clean rebuild
    rm -rf ~/.aiter/build/*

    # Copy the specific kernel version to the working directory
    cp ~/Grok/aiter/csrc/cpp_itfs/pa/Jacob/pa_kernels_${seg}.cuh ~/Grok/aiter/csrc/cpp_itfs/pa/pa_kernels.cuh

    # Run the first benchmark to warm up the GPU, without profiling
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 \
    python -m sglang.bench_one_batch --batch-size $bs --input 256 --output 2048 --tp 8 \
        --quantization fp8 --trust-remote-code --attention-backend aiter \
        --model $model_path  \
        --tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer

    # Run the second benchmark with profiling enabled
    RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 \
    python -m sglang.bench_one_batch --batch-size $bs --input 256 --output 2048 --tp 8 \
        --quantization fp8 --trust-remote-code --attention-backend aiter \
        --model $model_path  \
        --tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer \
        --enable-decode-prof

    sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv"   "select * from top;" ".output stdout"
    mv trace.rpd $out_folder/trace_bs${bs}_i256_o2048_${seg}.rpd
    mv trace.csv $out_folder/trace_bs${bs}_i256_o2048_${seg}.csv
done

# ---
### Reset to Original State

# Reset the kernel file to the original version
cp ~/Grok/aiter/csrc/cpp_itfs/pa/Jacob/pa_kernels_ori.cuh ~/Grok/aiter/csrc/cpp_itfs/pa/pa_kernels.cuh

# Run one final benchmark with the original kernel
RCCL_MSCCL_ENABLE=0 SGLANG_USE_AITER=1 SGLANG_INT4_WEIGHT=1 \
python -m sglang.bench_one_batch --batch-size $bs --input 256 --output 2048 --tp 8 \
    --quantization fp8 --trust-remote-code --attention-backend aiter \
    --model $model_path   \
    --tokenizer-path /data/huggingface/hub/Xenova/grok-1-tokenizer 