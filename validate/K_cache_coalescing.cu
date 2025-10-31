#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
using cache_t = __hip_bfloat16;

#define CHECK_HIP_ERROR(call)                                                \
    do {                                                                     \
        hipError_t err = call;                                               \
        if (err != hipSuccess) {                                             \
            std::cerr << "HIP error in " << #call << ": " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

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

__device__ __forceinline__ _B16x8 load_ntmprl_16Byte(const _B16x8* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto dat2       = loadnt(addr_alias + 2);
    auto dat3       = loadnt(addr_alias + 3);
    auto res        = make_float4(dat0, dat1, dat2, dat3);
    return *reinterpret_cast<_B16x8*>(&res);
}

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

// ---------------------- Kernels ----------------------

// 模擬 strided thread pattern (e.g., thread 0,16,32,...)
__global__ void kernel_strided(const cache_t* __restrict__ k_ptr,
                               const int* __restrict__ kphys_block,
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
    constexpr int QKHELOOP = head_dim / QKHE_PER_FETCH;
    

    // Get K block index
    // thread [0, 16, 32, 48], kphysical_block_number saves [0, 16, 32, 48]
    // thread [1, 17, 33, 49], kphysical_block_number saves [1, 17, 33, 49]
    // ...
    int kphysical_block_number[TLOOP];
    for(int token_depth = 0; token_depth < TLOOP; token_depth++) // 4
    {
        const int k_batch_offset = batch_idx * context_len;
        const int k_block_offset = partition_idx * T_PAR_SIZE;
        const int k_warp_offset = TOKENS_PER_WARP * warpid;
        const int k_thread_offset = token_depth * 16 + lane16id;
        const int k_token_idx  = k_batch_offset + k_block_offset + k_warp_offset + k_thread_offset;
        kphysical_block_number[token_depth] = kphys_block[k_token_idx];
        
        // if(blockIdx.x==0 && blockIdx.y ==0 && threadIdx.x<16)
        //     printf("threadIdx.x=%d, k_token_idx=%d\n",threadIdx.x, k_token_idx);
    }

    // Get K cache: 4 threads takes QKHELOOP iterations to load a full 256 bytes for a complete K cache. Load 4 K caches.
    // thread [0, 16, 32, 48]:
    //      for loop in K token [0, 16, 32, 48]: 
    //          for loop in QKHELOOP(0~3): // Each token head dim has 128 elements = 256 bytes
    //              thread  0 loads elements QKHELOOP*32 +  0~ 8
    //              thread 16 loads elements QKHELOOP*32 +  8~16
    //              thread 32 loads elements QKHELOOP*32 + 16~24
    //              thread 32 loads elements QKHELOOP*32 + 24~32
    // thread [1, 17, 33, 49]: ...
    _B16x8 Klocal[TLOOP][QKHELOOP];
    const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD; // thread [0, 16, 32, 48] --> row_head_elem [0, 8, 16, 24]
    for(int token_depth = 0; token_depth < TLOOP; token_depth++) // 4, 4 thread load 4 token
    {
        const int64_t kblock_number = static_cast<int64_t>(kphysical_block_number[token_depth]);
        int offset = kblock_number * head_dim;
        const cache_t* k_ptr2       = k_ptr + offset;

        for(int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) // 4, each thread load 16byte, so 4t load 64B, 4 iter load 256 Byte which match head_dim=128 elems(256B)
        {
            const int head_elem           = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
            const cache_t* k_fetch_ptr    = k_ptr2 + head_elem;
            const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
            _B16x8* out_16B = reinterpret_cast<_B16x8*>(out + offset + head_elem);
        
            // Klocal[token_depth][qkhe_depth] = *k_fetch_ptr_16B;
            *(out_16B) = load_ntmprl_16Byte(k_fetch_ptr_16B);
            // if (lane16id==0 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            //     printf("[Load K cache] threadIdx=%3d, token_depth=%3d, qkhe_depth=%3d, kblock_number=%3d, k_ptr=%llu "
            //             "k_ptr2=%llu, k_fetch_ptr_16B=%llu\n",
            //         threadIdx.x, token_depth, qkhe_depth, kblock_number, k_ptr, k_ptr2, k_fetch_ptr_16B); 
            
            // Check data is copied.
            // float test = (float)*(out + offset + head_elem);
            // float golden = (float)(*k_fetch_ptr);
            // if (threadIdx.x==0 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            //     printf("[Test] test=%f, golden=%f \n",test, golden);
        }
    }
}

// 模擬 coalesced thread pattern (thread 0~255 連續)
__global__ void kernel_coalesced_4t(const cache_t* __restrict__ Kcache,
                                 const int* __restrict__ kphys_block,
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
    constexpr int QKHELOOP = head_dim / QKHE_PER_FETCH;

    // Load K block idx 
    // Thread (1-based) 1,2,3,4 stores token idx 1, 17, 33, 49 
    // Thread (1-based) 5,6,7,8 stores token idx 2, 18, 34, 50
    int kphysical_blocks[TLOOP];
    for(int token_depth=0; token_depth<TLOOP; ++token_depth){
        const int seq_offset = blockIdx.x * context_len;
        const int block_offset = blockIdx.y * T_PAR_SIZE;
        const int warp_offset = warpid * WARP_SIZE;
        const int thread_offset = (threadIdx.x%64)/4 + token_depth * 16;
        const int idx = seq_offset +  block_offset + warp_offset + thread_offset;
        kphysical_blocks[token_depth] = kphys_block[idx];
    }

    // Load K cache
    // Thread (1-based) 1,2,3,4 stores token idx 1, 17, 33, 49 
    // Note: 為甚麼lds_buf的維度是64x32而不是256(token)x128(head_dim)?因為在QKHELOOP的迴圈，每四個thread一起讀64bytes=32elements，
    //       此時只會用到64個row，每個row只有用到32elements。若需要用到double buffer,則lds_buf[2][64][32]
    //       需考量:
    //       若一次讀256bytes的效率>32bytes，讓thread1~16讀token 1, thread17~32讀token 2...thread241~256讀token 16
    //       此時lds_buf維度是16x128, double buffer的維度 2x16x128
    // 這裡先用4個thread一組去讀K cache,LDS=64x32
     __shared__ cache_t lds_buf[64][32]; 
     const int rowId = threadIdx.x / 4; // range: 0~63
    for(int token_depth=0; token_depth<TLOOP; ++token_depth){
        const int physical_idx = kphysical_blocks[token_depth];
        const cache_t* k_ptr = Kcache + physical_idx * head_dim;
        for(int qkhe_loop=0; qkhe_loop<QKHELOOP; ++qkhe_loop){
            const int qk_offset = qkhe_loop * QKHE_PER_FETCH;
            const int thread_offset = threadIdx.x%4 * 8;
            const int offset = qk_offset + thread_offset;
            const _B16x8* ptr16 = reinterpret_cast<const _B16x8*>(k_ptr + offset);
            
            // *reinterpret_cast<_B16x8*>(&lds_buf[rowId][offset]) = load_ntmprl_16Byte(ptr16);

            // if (threadIdx.x<8 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            //     printf("[Load K cache: coalesce] threadIdx=%3d, token_depth=%3d, qkhe_depth=%3d, "
            //             "thread_offset=%3d, ptr16=%llu \n",
            //         threadIdx.x, token_depth, qkhe_loop, thread_offset, ptr16); 
            


            // write to output
            _B16x8* out_ptr16 = reinterpret_cast<_B16x8*>(out + physical_idx * head_dim + offset);
            *out_ptr16 = load_ntmprl_16Byte(ptr16);
            // *out_ptr16 = *reinterpret_cast<_B16x8*>(&lds_buf[rowId][offset]);
            
        }
        // if (threadIdx.x==0 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        //     printf("\n");
    }
}


// Remove block idx loading
// __attribute__((amdgpu_waves_per_eu(1, 3)))       
// __global__ __launch_bounds__(256) void kernel_coalesced_16t(
__global__ void kernel_coalesced_16t(
    const cache_t* __restrict__ Kcache,
    const int* __restrict__ kphys_block,
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
    const int n_thread_per_group = 16;
    const int n_group = WARP_SIZE / n_thread_per_group; //  4
    const int TLOOP_16t  = TOKENS_PER_WARP / n_group;   // 16
    const int QKHE_PER_FETCH_16t = 128;
    constexpr int QKHELOOP_16t = head_dim / QKHE_PER_FETCH_16t; // 1

    // Load K block idx 
    // Thread (1-based)  1~16 stores token idx 1, 5,  9, 13, 17, 21...61
    // Thread (1-based) 17~32 stores token idx 2, 6, 10, 14, 18, 22...62
    // Thread (1-based) 33~48 stores token idx 3, 7, 11, 15, 19, 23...63
    // Thread (1-based) 49~64 stores token idx 4, 8, 12, 16, 20, 24...64
    int kphysical_blocks[TLOOP_16t];
    for(int token_depth=0; token_depth<TLOOP_16t; ++token_depth){
        const int seq_offset = blockIdx.x * context_len;
        const int block_offset = blockIdx.y * T_PAR_SIZE;
        const int warp_offset = warpid * WARP_SIZE;
        const int thread_offset = token_depth * n_group + (threadIdx.x%64) / n_thread_per_group;
        const int idx = seq_offset +  block_offset + warp_offset + thread_offset;
        kphysical_blocks[token_depth] = kphys_block[idx];
    }

    // Load K cache
    // 這裡用16個thread去讀K cache,LDS=16x128 = 4KB
    __shared__ cache_t lds_buf[16][128]; 
    const int token_idx = threadIdx.x / 16; // range: 0~16
    for(int token_depth=0; token_depth<TLOOP_16t; ++token_depth){
        const int physical_idx = kphysical_blocks[token_depth];
        const cache_t* k_ptr = Kcache + physical_idx * head_dim;
        for(int qkhe_loop=0; qkhe_loop<QKHELOOP_16t; ++qkhe_loop){
            const int qk_offset = qkhe_loop * QKHE_PER_FETCH_16t;
            const int thread_offset = threadIdx.x%16 * 8;
            const int offset = qk_offset + thread_offset;
            const _B16x8* ptr16 = reinterpret_cast<const _B16x8*>(k_ptr + offset);
            

            // if (threadIdx.x<16 && warpid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            //     printf("[Load K cache: 16t] threadIdx=%3d, token_depth=%3d, qkhe_depth=%3d, "
            //             "thread_offset=%3d, ptr16=%llu \n",
            //         threadIdx.x, token_depth, qkhe_loop, thread_offset, ptr16); 

            // write to output
            _B16x8* out_ptr16 = reinterpret_cast<_B16x8*>(out + physical_idx * head_dim + offset);
            *out_ptr16 = load_ntmprl_16Byte(ptr16);

            // in global --> LDS --> out global
            // *reinterpret_cast<_B16x8*>(&lds_buf[token_idx][offset]) = load_ntmprl_16Byte(ptr16);
            // *out_ptr16 = *reinterpret_cast<_B16x8*>(&lds_buf[token_idx][offset]);
            


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
    int* h_phys;
    cache_t* h_out_strided       = (cache_t*)malloc(sizeof(cache_t) * K_size);
    cache_t* h_out_coalesced_4t  = (cache_t*)malloc(sizeof(cache_t) * K_size);
    cache_t* h_out_coalesced_16t = (cache_t*)malloc(sizeof(cache_t) * K_size);

    // Fill data
    for (size_t i = 0; i < K_size; ++i)
        h_K[i] = (uint16_t)(i & 0xFFFF);
    h_phys = generate_unique_random_permutation(phys_size);

    // for(int i=0;i <5;++i)  printf("h_phys[%d]=%d\n",i ,h_phys[i]);

    // Allocate device memory
    cache_t* d_K;
    int* d_phys;
    cache_t* d_out_strided;
    cache_t* d_out_coalesced_4t;
    cache_t* d_out_coalesced_16t;

    hipMalloc(&d_K, sizeof(cache_t) * K_size);
    hipMalloc(&d_phys, sizeof(int) * phys_size);
    hipMalloc(&d_out_strided, sizeof(cache_t) * K_size);
    hipMalloc(&d_out_coalesced_4t, sizeof(cache_t) * K_size);
    hipMalloc(&d_out_coalesced_16t, sizeof(cache_t) * K_size);

    hipMemcpy(d_K, h_K, sizeof(cache_t) * K_size, hipMemcpyHostToDevice);
    hipMemcpy(d_phys, h_phys, sizeof(int) * phys_size, hipMemcpyHostToDevice);

    // Launch
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(n_threads_in_block, 1, 1);

    // warmup
    for(int i=0; i<5; ++i){
        hipLaunchKernelGGL(kernel_strided, grid, block, 0, 0, d_K, d_phys, d_out_strided);
        hipLaunchKernelGGL(kernel_coalesced_4t, grid, block, 0, 0, d_K, d_phys, d_out_coalesced_4t);
        hipLaunchKernelGGL(kernel_coalesced_16t, grid, block, 0, 0, d_K, d_phys, d_out_coalesced_16t);
    }
    hipLaunchKernelGGL(kernel_strided, grid, block, 0, 0, d_K, d_phys, d_out_strided);
    hipLaunchKernelGGL(kernel_coalesced_4t, grid, block, 0, 0, d_K, d_phys, d_out_coalesced_4t);
    hipLaunchKernelGGL(kernel_coalesced_16t, grid, block, 0, 0, d_K, d_phys, d_out_coalesced_16t);
    hipDeviceSynchronize();

    printf("Done.\n");

    // ToDo: Compare values with d_K
    CHECK_HIP_ERROR(hipMemcpy(h_out_strided, d_out_strided, sizeof(cache_t) * K_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_out_coalesced_4t, d_out_coalesced_4t, sizeof(cache_t) * K_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_out_coalesced_16t, d_out_coalesced_16t, sizeof(cache_t) * K_size, hipMemcpyDeviceToHost));    
    bool correct_strided = true;
    for(int i=0; i<K_size; ++i)
        if(h_out_strided[i]!=h_K[i]){
            printf("[ERROR: kernel_strided] h_out_strided[%d]!=h_K[%d], %f %f\n",
                i, i, (float)h_out_strided[i], (float)h_K[i]);
            correct_strided = false;
            break;
        }
    if(correct_strided) printf("kernel_strided pass numerical check.\n");
    
    bool correct_coalesced_4t = true;
    for(int i=0; i<K_size; ++i)
        if(h_out_coalesced_4t[i]!=h_K[i]){
            printf("[ERROR: h_out_coalesced] h_out_coalesced_4t[%d]!=h_K[%d], %f %f\n",
                i, i, (float)h_out_coalesced_4t[i], (float)h_K[i]);
            correct_coalesced_4t = false;
            break;
        }
    if(correct_coalesced_4t) printf("h_out_coalesced_4t pass numerical check.\n");

    bool correct_coalesced_16t = true;
    for(int i=0; i<K_size; ++i)
        if(h_out_coalesced_16t[i]!=h_K[i]){
            printf("[ERROR: h_out_coalesced] h_out_coalesced_16t[%d]!=h_K[%d], %f %f\n",
                i, i, (float)h_out_coalesced_16t[i], (float)h_K[i]);
            correct_coalesced_16t = false;
            break;
        }
    if(correct_coalesced_16t) printf("h_out_coalesced_16t pass numerical check.\n");
    

    // for(int i=0;i<5;++i){
    //     float test = (float)h_out_coalesced_16t_no[i];
    //     float gold = (float)h_K[i];
    //     printf("%f %f \n", test, gold);
    // }

    // cleanup
    hipFree(d_K);
    hipFree(d_phys);
    hipFree(d_out_strided);
    hipFree(d_out_coalesced_4t);
    hipFree(d_out_coalesced_16t);
    free(h_K);
    free(h_phys);
    return 0;
}
