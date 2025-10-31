#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

#define CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

constexpr int THREADS_PER_BLOCK = 256;
constexpr int N = 1 << 24;  // 16M elements (64 MB)

// ---------------- Kernels ----------------
__global__ void copy_float(const float* __restrict__ src, float* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

__global__ void copy_float2(const float2* __restrict__ src, float2* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2)
        dst[idx] = src[idx];
}

__global__ void copy_float4(const float4* __restrict__ src, float4* __restrict__ dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4)
        dst[idx] = src[idx];
}

// ---------------- Alignment helper ----------------
void* hipMallocAligned(size_t bytes, size_t alignment)
{
    // hipMalloc guarantees at least 256B alignment, but we ensure stricter alignment here
    void* ptr;
    CHECK(hipMalloc(&ptr, bytes + alignment));

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + (alignment - 1)) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);

    // Sanity check
    if (reinterpret_cast<uintptr_t>(aligned_ptr) % alignment != 0) {
        std::cerr << "Alignment failed!" << std::endl;
        exit(1);
    }

    std::cout << "Allocated base: " << ptr
              << ", aligned: " << aligned_ptr
              << ", alignment = " << alignment << " bytes\n";

    return aligned_ptr;
}

// ---------------- Host main ----------------
int main()
{
    size_t bytes = N * sizeof(float);
    float *h_src = new float[N];
    for (int i = 0; i < N; ++i) h_src[i] = float(i % 123);

    // Allocate aligned device memory
    float* d_src = (float*)hipMallocAligned(bytes, 16);
    float* d_dst = (float*)hipMallocAligned(bytes, 16);

    // Copy to device
    CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid_float((N + block.x - 1) / block.x);
    dim3 grid_float2((N / 2 + block.x - 1) / block.x);
    dim3 grid_float4((N / 4 + block.x - 1) / block.x);

    // Warmup
    hipLaunchKernelGGL(copy_float, grid_float, block, 0, 0, d_src, d_dst);
    hipLaunchKernelGGL(copy_float2, grid_float2, block, 0, 0,
                       reinterpret_cast<const float2*>(d_src),
                       reinterpret_cast<float2*>(d_dst));
    hipLaunchKernelGGL(copy_float4, grid_float4, block, 0, 0,
                       reinterpret_cast<const float4*>(d_src),
                       reinterpret_cast<float4*>(d_dst));
    CHECK(hipDeviceSynchronize());

    // --- Benchmark float ---
    auto t1 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(copy_float, grid_float, block, 0, 0, d_src, d_dst);
    CHECK(hipDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_float = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // --- Benchmark float4 ---
    auto t3 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(copy_float4, grid_float4, block, 0, 0,
                       reinterpret_cast<const float4*>(d_src),
                       reinterpret_cast<float4*>(d_dst));
    CHECK(hipDeviceSynchronize());
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_float4 = std::chrono::duration<double, std::milli>(t4 - t3).count();

    double gb = bytes / 1e9;
    std::cout << "\n==== Bandwidth Test ====\n";
    std::cout << "float  copy:  " << gb / (ms_float / 1e3) << " GB/s\n";
    std::cout << "float4 copy:  " << gb / (ms_float4 / 1e3) << " GB/s\n";

    hipFree(d_src);
    hipFree(d_dst);
    delete[] h_src;
}
