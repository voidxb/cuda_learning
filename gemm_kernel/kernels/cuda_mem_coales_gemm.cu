#include <cuda_bf16.h>
#include <cuda_runtime.h>


template<typename T>
__global__ void mem_coalesced_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
) {

    int block_size = 32;
    const int x = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int y = blockIdx.y * block_size + (threadIdx.x % block_size);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += float(A[x * K + i]) * float(B[i * N + y]);
        }
            C[x * N + y] = T(tmp);
    }
}

// 主机端调用函数
template<typename T>
void mem_coalesced_gemm(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
) {

    dim3 threadsPerBlock(32 * 32);
    dim3 numBlocks(
        CEIL_DIV(M, 32),
        CEIL_DIV(N, 32)
    );
    
    mem_coalesced_gemm_kernel<T><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
