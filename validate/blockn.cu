#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono> 
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
int* generate_unique_random_permutation(size_t num_page_blocks) {
    if (num_page_blocks == 0) {
        return nullptr;
    }

    // 1. 使用 std::vector 作為臨時容器，因為它支持 std::shuffle
    std::vector<int> temp_phys(num_page_blocks);

    // 2. 填充 temp_phys 陣列，使其包含 0 到 num_page_blocks-1 的連續值
    for (size_t i = 0; i < num_page_blocks; ++i) {
        temp_phys[i] = static_cast<int>(i);
    }

    // 3. 設置隨機數生成器
    //    使用時間作為種子來確保每次執行都有不同的排列
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);

    // 4. 使用 std::shuffle 進行亂序排列 (確保不重複)
    std::shuffle(temp_phys.begin(), temp_phys.end(), rng);

    // 5. 使用 malloc 分配最終的 int 陣列
    int* h_phys = (int*)malloc(sizeof(int) * num_page_blocks);
    if (h_phys == nullptr) {
        std::cerr << "Error: malloc failed for h_phys" << std::endl;
        return nullptr;
    }

    // 6. 將亂序排列的結果複製到 malloc 分配的陣列中
    std::copy(temp_phys.begin(), temp_phys.end(), h_phys);
    
    // 返回 malloc 分配的陣列，呼叫者需負責 free
    return h_phys;
}


using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x8 _B16x8;


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
constexpr int BlockSize = 256; // Page block size
constexpr int n_threads_in_block = 256;
constexpr int T_PAR_SIZE = 256;     // Each block(256 threads) process 256 tokens
constexpr int WARP_SIZE = 64;
constexpr int NWARPS = 4;
constexpr int TOKENS_PER_WARP = 64; // Each warp process 64 tokens
constexpr int ELEMS_PER_THREAD = sizeof(_B16x8) / sizeof(cache_t); // Each thread load 8 elements
constexpr int BYTES_PER_FETCH = WARP_SIZE * 16; // 1024 bytes
constexpr int TOKEN_PER_FETCH = BYTES_PER_FETCH / (head_dim * sizeof(cache_t)); // 4 token
constexpr int TLOOP = TOKENS_PER_WARP / TOKEN_PER_FETCH; // 16
constexpr int TOKEN_PER_WG = TOKEN_PER_FETCH * NWARPS; // 16 tokens per workgroup 
constexpr int max_num_blocks_per_seq = (context_len + BlockSize - 1) / BlockSize;
constexpr int num_page_blocks = max_num_blocks_per_seq * bs;
constexpr int kv_block_stride = BlockSize * head_dim;
constexpr int N_THREAD_LOAD_TOKEN = head_dim / ELEMS_PER_THREAD; // if head_dim=128, bf16 --> 16 threads load 1 token

// ---------------------- Kernels ----------------------
__global__ void kernel_coalesced(
    const int* __restrict__ KBlockIdx,
    cache_t* __restrict__ Kcache,
    cache_t* __restrict__ out)
{
    const int batch_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warpid     = threadIdx.x / WARP_SIZE;
    const int laneid     = threadIdx.x % WARP_SIZE;
    const int rowid      = laneid / N_THREAD_LOAD_TOKEN; // Every 16 threads load 1 token in 1 TLOOP iteration
    const int lane16id   = laneid % N_THREAD_LOAD_TOKEN; // In each 16 threads, get its index 0~15

    // Load K block idx 
    // (1-based)  1st wavefront stores token idx   1~4
    // (1-based)  2nd wavefront stores token idx   5~8
    // (1-based)  3rd wavefront stores token idx  9~12
    // (1-based)  4th wavefront stores token idx 13~16 ...
    int kphysical_blocks[TLOOP], kphysical_offset[TLOOP];
    for(int token_depth=0; token_depth<TLOOP; ++token_depth){
        const int seq_offset = batch_idx * context_len;
        const int block_offset = blockIdx.y * T_PAR_SIZE;
        const int warp_offset = token_depth * TOKEN_PER_WG + warpid * TOKEN_PER_FETCH;
        const int thread_offset = rowid;
        const int token_idx = seq_offset +  block_offset + warp_offset + thread_offset;
        const int page_idx = token_idx / BlockSize;
        const int page_offset = token_idx % BlockSize;
        kphysical_blocks[token_depth] = KBlockIdx[page_idx];
        kphysical_offset[token_depth] = page_offset;


        // if (threadIdx.x<64 && token_depth==0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        //     printf("[K block] threadIdx=%3d, token_depth=%3d, kphys_block[idx]=%6d offset=%d\n",
        //         threadIdx.x, token_depth, KBlockIdx[idx], offset); 
    }

    // Load K cache
    __shared__ cache_t lds_buf[16][128]; 
    
    const int token_idx = threadIdx.x / 16; // range: 0~16
    for(int token_depth=0; token_depth<TLOOP; ++token_depth){
        const int kblock_number = kphysical_blocks[token_depth];
        const int kblock_offset = kphysical_offset[token_depth];
        // const int global_offset = kblock_number * kv_block_stride + kblock_offset * head_dim;
        // const int thread_offset = lane16id * ELEMS_PER_THREAD;
        // const int offset = global_offset + thread_offset;
        const int offset = kblock_number * kv_block_stride + kblock_offset * head_dim + lane16id * ELEMS_PER_THREAD;
        const _B16x8* k_ptr_B16x8 = reinterpret_cast<const _B16x8*>(Kcache + offset);
        _B16x8* out_ptr16 = reinterpret_cast<_B16x8*>(out + offset);

        // write to output
        *out_ptr16 = load_ntmprl_16Byte(k_ptr_B16x8);

        // float out_fp16 = (float)out[2048];
        // if (out_fp16!=0) printf("out[2048]=%f\n", out_fp16);
        
        // LDS
        // *reinterpret_cast<_B16x8*>(&lds_buf[token_idx][lane16id * ELEMS_PER_THREAD]) = load_ntmprl_16Byte(k_ptr_B16x8);
        // __syncthreads();
        // *out_ptr16 = *reinterpret_cast<_B16x8*>(&lds_buf[token_idx][lane16id * ELEMS_PER_THREAD]);
        
        // if (threadIdx.x<64 && token_depth==0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        //     printf("[K block] threadIdx=%3d, token_depth=%3d, kblock_offset=%d, k_ptr_B16x8=%llu "
        //             "global_offset=%d, thread_offset=%d, offset=%d\n",
        //         threadIdx.x, token_depth, kblock_offset, k_ptr_B16x8, 
        //         global_offset, thread_offset, offset); 
    }
}

// ---------------------- Host ----------------------
int main() {
    
    size_t K_cache_size = (size_t)bs * context_len * head_dim;    

    // Allocate host memory
    int* h_KBlockIdx = generate_unique_random_permutation(num_page_blocks);
    cache_t* h_KCache = (cache_t*)malloc(sizeof(cache_t) * K_cache_size);
    cache_t* h_out_coalesced = (cache_t*)malloc(sizeof(cache_t) * K_cache_size);

    // Fill data
    unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    std::uniform_int_distribution<uint16_t> distribution(
        std::numeric_limits<uint16_t>::min(), 
        std::numeric_limits<uint16_t>::max()
    );

    for (size_t i = 0; i < K_cache_size; ++i) {
        h_KCache[i] = distribution(generator);
    }
    // for (size_t i = 0; i < K_cache_size; ++i)
    //     h_KCache[i] = (uint16_t)(i & 0xFFFF);


    // Allocate device memory
    int* d_KBlockIdx;
    cache_t* d_KCache;
    cache_t* d_out_coalesced;

    hipMalloc(&d_KBlockIdx, sizeof(int) * num_page_blocks);
    hipMalloc(&d_KCache,        sizeof(cache_t) * K_cache_size);
    hipMalloc(&d_out_coalesced, sizeof(cache_t) * K_cache_size);

    hipMemcpy(d_KBlockIdx, h_KBlockIdx, sizeof(int) * num_page_blocks, hipMemcpyHostToDevice);
    hipMemcpy(d_KCache,    h_KCache,    sizeof(cache_t) * K_cache_size     , hipMemcpyHostToDevice);

    // Launch
    constexpr int grid_x = bs;
    constexpr int grid_y = context_len / n_threads_in_block;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(n_threads_in_block, 1, 1);

    printf("BlockSize=%d\n", BlockSize);
    printf("Warmup...\n");
    for(int i=0; i<5; ++i){// warmup
        kernel_coalesced<<<grid, block>>>(d_KBlockIdx, d_KCache, d_out_coalesced);
    }
    kernel_coalesced<<<grid, block>>>(d_KBlockIdx, d_KCache, d_out_coalesced);
    hipDeviceSynchronize();

    printf("Done.\n");

    // Compare values with d_K
    CHECK_HIP_ERROR(hipMemcpy(h_out_coalesced, d_out_coalesced, sizeof(cache_t) * K_cache_size, hipMemcpyDeviceToHost));
  
    bool correct_coalesced = true;
    int n_mismtach = 0;
    for(int i=0; i<K_cache_size; ++i)
        if(h_out_coalesced[i]!=h_KCache[i]){
            printf("[ERROR: h_out_coalesced] h_out_coalesced[%d]!=h_KCache[%d], %f %f\n",
                i, i, (float)h_out_coalesced[i], (float)h_KCache[i]);
            correct_coalesced = false;
            n_mismtach++;
            if(n_mismtach>=5)
                break;
        }
    if(correct_coalesced) printf("h_out_coalesced pass numerical check.\n");
    
    return 0;
}
