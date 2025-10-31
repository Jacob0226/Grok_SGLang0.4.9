# GPU Memory Coalescing Profiling and Analysis for K block/cache and BlockSize

This document provides compilation, execution, and profiling steps for several HIP-based GPU kernels used to study **memory coalescing** behavior on AMD CDNA4 architecture using **rocprof-compute** and **rocprofv3**.

---

## 🧹 Clean Previous Builds

```bash
sudo rm ~/.aiter/build/pa*
```

🔹 1. K_cache_coalescing  
3 kernels: 
1. 4 strided threads load continuous 4x16 bytes
2. 4 continuous threads load continuous 4x16 bytes
3. 16 continuous threads load continuous 16x16 bytes
```bash
# Build
mkdir K_cache_coalescing
hipcc K_cache_coalescing.cu -o K_cache_coalescing/K_cache_coalescing.out \
    --save-temps -mllvm --amdgpu-kernarg-preload-count=16 \
    -O3 -std=c++17 -DENABLE_FP8 -fgpu-flush-denormals-to-zero

# Run
~/Grok_SGLang0.4.9/validate/K_cache_coalescing/K_cache_coalescing.out

# Profile
export PATH=/app/rocm-systems/projects/rocprofiler-compute/install/bin:$PATH
export ROCPROF=rocprofiler-sdk
export KERNEL=kernel_coalesced_16t # choice: kernel_strided, kernel_coalesced_4t, kernel_coalesced_16t 
export OUT=kernel_coalesced_16t_1030
export HIP_VISIBLE_DEVICES=5
rocprof-compute profile -n ${OUT} -k ${KERNEL} --no-roof \
    -- ~/Grok_SGLang0.4.9/validate/K_cache_coalescing/K_cache_coalescing.out

# Analyze
rocprof-compute analyze -p workloads/${OUT}/MI355/ -d 5 -t us -b 0 2.1 16.1.3

# ATT
export Mode=ATT
export KERNEL=kernel_coalesced_16t
export HIP_VISIBLE_DEVICES=7
export ROCPROF_ATT_LIBRARY_PATH=/app/TT/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux/opt/rocm/lib/
rocprofv3 --att -d ${Mode}_${KERNEL} --att-target-cu 0 \
    --kernel-include-regex $KERNEL --kernel-iteration-range 6 \
    --att-activity 10 \
    -- ~/Grok_SGLang0.4.9/validate/a.out
```
---
🔹 2. coalesce_fp32.cu  
Discard vector load of bit16x8 and use single fp32 to observe coalescing rate reaching 100% with wavefront loading conti. memory.
```bash
# Build
hipcc coalesce_fp32.cu -o coalesce_fp32/coalesce_fp32.out \
    --save-temps -mllvm --amdgpu-kernarg-preload-count=16 \
    -O3 -std=c++17 -DENABLE_FP8 -fgpu-flush-denormals-to-zero
~/Grok_SGLang0.4.9/validate/coalesce_fp32/coalesce_fp32.out

# Profile
export PATH=/app/rocm-systems/projects/rocprofiler-compute/install/bin:$PATH
export ROCPROF=rocprofiler-sdk
export KERNEL=kernel_coalesced_64t
export OUT=1030
export HIP_VISIBLE_DEVICES=4
rocprof-compute profile -n ${OUT} -k ${KERNEL} --no-roof \
    -- ~/Grok_SGLang0.4.9/validate/coalesce_fp32/coalesce_fp32.out

# Analyze
rocprof-compute analyze -p workloads/kernel_coalesced_64t_NT/MI355/ -d 5 -t us -b 0 2.1 16.1.3
```
---
🔹 3. float vs float2 vs float4  
Surprisingly, float2 and float4 only reach 25% coalescing access
```bash
# Build
hipcc float_vs_float4.cu -o float_vs_float4/float_vs_float4.out \
    --save-temps -mllvm --amdgpu-kernarg-preload-count=16 \
    -O3 -std=c++17 -DENABLE_FP8 -fgpu-flush-denormals-to-zero

# Profile
export PATH=/app/rocm-systems/projects/rocprofiler-compute/install/bin:$PATH
export ROCPROF_ATT_LIBRARY_PATH=/app/TT/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux/opt/rocm/lib/
export ROCPROF=rocprofiler-sdk
export OUT=float_vs_float2_vs_float4_aligned
export HIP_VISIBLE_DEVICES=4
rocprof-compute profile -n ${OUT} --no-roof \
    -- ~/Grok_SGLang0.4.9/validate/float_vs_float4/float_vs_float4.out

# Analyze
rocprof-compute analyze -p workloads/${OUT}/MI355/ -d 1 -t us -b 0 2.1 16.1.3
```
---
🔹 4. blockn.cu (IMPORTANT)  
This implementation speeds up the kernel.
Take Grok1 for the example, head_dim=128, cache type = bf16. 1 token K cache = 128*2 bytes = 256 bytes
Each wavefront loads the continuous 1024 bytes with block_size=[16, 64]
```bash
# Build
hipcc blockn.cu -o blockn/blockn.out \
    --save-temps -mllvm --amdgpu-kernarg-preload-count=16 \
    -O3 -std=c++17 -DENABLE_FP8 -fgpu-flush-denormals-to-zero

# Profile
export PATH=/app/rocm-systems/projects/rocprofiler-compute/install/bin:$PATH
export ROCPROF=rocprofiler-sdk
export KERNEL=kernel_coalesced
export OUT=PageSize256
export HIP_VISIBLE_DEVICES=4
rocprof-compute profile -n ${OUT} -k ${KERNEL} --no-roof \
    -- ~/Grok_SGLang0.4.9/validate/blockn/blockn.out

# Analyze
rocprof-compute analyze -p workloads/${OUT}/MI355/ -d 5 -t us -b 0 2.1 16.1.3
```
