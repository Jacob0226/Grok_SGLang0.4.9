#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
using cache_t =  float;

#define CHECK_HIP_ERROR(call)                                                \
    do {                                                                     \
        hipError_t err = call;                                               \
        if (err != hipSuccess) {                                             \
            std::cerr << "HIP error in " << #call << ": " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
    _B16x4 xy[2];
} _B16x8;

template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

// __device__ __forceinline__ _B16x8 load_ntmprl_16Byte(const _B16x8* addr)
// {
//     auto addr_alias = reinterpret_cast<const float*>(addr);
//     auto dat0       = loadnt(addr_alias);
//     auto dat1       = loadnt(addr_alias + 1);
//     auto dat2       = loadnt(addr_alias + 2);
//     auto dat3       = loadnt(addr_alias + 3);
//     auto res        = make_float4(dat0, dat1, dat2, dat3);
//     return *reinterpret_cast<_B16x8*>(&res);
// }

// ---------------------- Constants ----------------------
constexpr int bs = 512;
constexpr int context_len = 2048; // Must be multiples of 256
constexpr int head_dim = 128;
constexpr int n_threads_in_block = 256;
constexpr int grid_x = bs;
constexpr int grid_y = context_len / n_threads_in_block;
constexpr int total_tokens = bs * context_len;
constexpr int loads_per_thread = head_dim * sizeof(cache_t) / sizeof(_B16x8); // 128 * 2 / 16 = 16
constexpr int WARP_SIZE = 64;
constexpr int QKHE_PER_FETCH = 32;
constexpr int TOKENS_PER_WARP = 64; // Each warp process 64 tokens
constexpr int T_PAR_SIZE = 256;     // Each block(256 threads) process 256 tokens
constexpr int TLOOP = TOKENS_PER_WARP / 16; 
constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD = 16 / sizeof(cache_t); // 8 for 16 bit cache type

// ---------------------- Utility ----------------------
__device__ inline _B16x8 load_16B(const _B16x8* ptr) {
    return *ptr; // 可換成 non-temporal intrinsic
}

// 函式：產生 0 到 size-1 的不重複隨機排列
int* generate_unique_random_permutation(size_t phys_size) {
    if (phys_size == 0) {
        return nullptr;
    }

    // 1. 使用 std::vector 作為臨時容器，因為它支持 std::shuffle
    std::vector<int> temp_phys(phys_size);

    // 2. 填充 temp_phys 陣列，使其包含 0 到 phys_size-1 的連續值
    for (size_t i = 0; i < phys_size; ++i) {
        temp_phys[i] = static_cast<int>(i);
    }

    // 3. 設置隨機數生成器
    //    使用時間作為種子來確保每次執行都有不同的排列
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);

    // 4. 使用 std::shuffle 進行亂序排列 (確保不重複)
    std::shuffle(temp_phys.begin(), temp_phys.end(), rng);

    // 5. 使用 malloc 分配最終的 int 陣列
    int* h_phys = (int*)malloc(sizeof(int) * phys_size);
    if (h_phys == nullptr) {
        std::cerr << "Error: malloc failed for h_phys" << std::endl;
        return nullptr;
    }

    // 6. 將亂序排列的結果複製到 malloc 分配的陣列中
    std::copy(temp_phys.begin(), temp_phys.end(), h_phys);
    
    // 返回 malloc 分配的陣列，呼叫者需負責 free
    return h_phys;
}

// ---------------------- Kernels ----------------------

// 64 threads coalesce access 64 float
__global__ void kernel_coalesced_64t(
    const int* __restrict__ kphys_block,
    const cache_t* __restrict__ k_ptr,
    cache_t* __restrict__ out)
{
    const int batch_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    int block_token = blockIdx.y * n_threads_in_block;
    const int warpid     = threadIdx.x / WARP_SIZE;
    const int laneid     = threadIdx.x % WARP_SIZE;
    const int lane4id    = laneid % 4;
    const int lane16id   = laneid % 16;
    const int rowid      = laneid / 16;
    const int n_thread_per_group = 64;
    const int n_group = WARP_SIZE / n_thread_per_group; //  1
    const int TLOOP_64t  = TOKENS_PER_WARP / n_group;   // 64
    const int QKHE_PER_FETCH_64t = 64;
    constexpr int QKHELOOP_64t = head_dim / QKHE_PER_FETCH_64t; // 2

    // Load K block idx 
    // Thread (1-based)   1~ 64 stores token idx    1~64
    // Thread (1-based)  65~128 stores token idx  65~128
    // Thread (1-based) 129~192 stores token idx 129~192
    // Thread (1-based) 193~256 stores token idx 129~192
    int kphysical_blocks[TLOOP_64t];
    for(int token_depth=0; token_depth<TLOOP_64t; ++token_depth){
        const int seq_offset = blockIdx.x * context_len;
        const int block_offset = blockIdx.y * T_PAR_SIZE;
        const int warp_offset = warpid * WARP_SIZE;
        const int thread_offset = token_depth;
        const int idx = seq_offset +  block_offset + warp_offset + thread_offset;
        kphysical_blocks[token_depth] = idx; //kphys_block[idx];
    }

    // Load K cache
    // 這裡用64個thread去讀K cache
    // __shared__ cache_t lds_buf[16][128]; 
    // const int token_idx = threadIdx.x / 16; // range: 0~16
    for(int token_depth=0; token_depth<TLOOP_64t; ++token_depth){
        const int physical_idx = kphysical_blocks[token_depth];
        const int token_offset = physical_idx * head_dim;
        for(int qkhe_loop=0; qkhe_loop<QKHELOOP_64t; ++qkhe_loop){
            const int qk_offset = qkhe_loop * QKHE_PER_FETCH_64t;
            const int thread_offset = threadIdx.x%64;
            const int offset = token_offset + qk_offset + thread_offset;
            // *(out + offset) = *(k_ptr + offset);
            // NT load
            *(out + offset) = loadnt(k_ptr + offset);

            // if (threadIdx.x==0 && token_depth <64 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            //     printf("[Load K cache: 64t] threadIdx=%3d, token_depth=%3d, qkhe_depth=%3d, "
            //             "physical_idx=%6d, offset=%3d, \n",
            //         threadIdx.x, token_depth, qkhe_loop, physical_idx, offset); 
        }
        // if (threadIdx.x==0 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        //     printf("\n");
    }
}


// ---------------------- Host ----------------------
int main() {
    size_t K_size = (size_t)bs * context_len * head_dim;
    size_t phys_size = (size_t)bs * context_len;

    // Allocate host memory
    cache_t* h_K = (cache_t*)malloc(sizeof(cache_t) * K_size);
    cache_t* h_out = (cache_t*)malloc(sizeof(cache_t) * K_size);
    int* h_phys = generate_unique_random_permutation(phys_size); 

    // Fill data
    for (size_t i = 0; i < K_size; ++i)
        h_K[i] = 5; //(int)(i & 0xFFFF);

    // Allocate device memory
    cache_t* d_K, *d_out;
    int* d_phys;
    hipMalloc(&d_K, sizeof(cache_t) * K_size);
    hipMalloc(&d_out, sizeof(cache_t) * K_size);
    CHECK_HIP_ERROR(hipMemset(d_out, 0, sizeof(cache_t) * K_size));
    hipMalloc(&d_phys, sizeof(int) * phys_size);
    hipMemcpy(d_K, h_K, sizeof(cache_t) * K_size, hipMemcpyHostToDevice);
    hipMemcpy(d_phys, h_phys, sizeof(int) * phys_size, hipMemcpyHostToDevice);

    // Launch
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(n_threads_in_block, 1, 1);

    // warmup
    for(int i=0; i<5; ++i){
        hipLaunchKernelGGL(kernel_coalesced_64t, grid, block, 0, 0, d_phys, d_K, d_out);
    }
    hipLaunchKernelGGL(kernel_coalesced_64t, grid, block, 0, 0, d_phys, d_K, d_out);
    hipDeviceSynchronize();

    printf("Done.\n");
    hipMemcpy(h_out, d_out, sizeof(int) * K_size, hipMemcpyDeviceToHost);

    // Check
    for(int i=0; i<K_size; ++i)
        if(h_out[i]!=5){
            printf("h_out=%f != 5, i=%d\n", h_out[i], i);
            break;
        }

    

    // cleanup
    hipFree(d_K);
    hipFree(d_out);
    free(h_K);
    return 0;
}
