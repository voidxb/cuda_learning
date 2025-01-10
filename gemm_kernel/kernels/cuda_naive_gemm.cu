#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "util.cuh"

template<typename T>
__global__ void naive_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
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
void naive_gemm(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
) {
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
       CEIL_DIV(M, 32),
       CEIL_DIV(N, 32)
    );
    
    naive_gemm_kernel<T><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}