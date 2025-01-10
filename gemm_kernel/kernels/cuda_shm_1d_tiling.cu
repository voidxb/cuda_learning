#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "util.cuh"

typedef unsigned int uint;  // 显式定义uint类型


template <const uint BLOCK_M, const uint BLOCK_N, const uint BLOCK_K, const uint TM, typename T>
__global__ void sgemm1DBlocktiling(const T *__restrict__ A,
                                   const T *__restrict__ B,
                                   T *__restrict__ C,
                                   const int M, const int N, const int K)
{
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    // 先在C中找到当前的块索引
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    // 因为BLOCK_N对应的是x维度，所以所有操作符都是x, 且该 kernel 只用 x 作为线程索引
    const int threadCol = threadIdx.x % BLOCK_N;
    const int threadRow = threadIdx.x / BLOCK_N;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BLOCK_M * BLOCK_K];
    __shared__ float Bs[BLOCK_K * BLOCK_N];

    // Move blocktile to beginning of A's row and B's column
    // A找起始点：每个BLOCK的size是BLOCK_M * BLOCK_K
    A += cRow * BLOCK_M * K;
    // B找起始点：每个BLOCK的size是BLOCK_N(row major, stride就是1)
    B += cCol * BLOCK_N;
    // C找起始点：需要先跳到A的行，再跳到B的列, 所以是两部分
    C += cRow * BLOCK_M * N + cCol * BLOCK_N;

    // todo: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    // assert(BLOCK_M * BLOCK_K == blockDim.x);
    // assert(BLOCK_N * BLOCK_K == blockDim.x);
    const uint innerColA = threadIdx.x % BLOCK_K; // warp-level GMEM coalescing，所有row的部分都是由x变化引起， 而col的部分可以缓慢变化（或者由y引起）
    const uint innerRowA = threadIdx.x / BLOCK_K; // 这里用BLOCK_K是因为A是往K轴延伸，K是row的维度
    const uint innerColB = threadIdx.x % BLOCK_N; // warp-level GMEM coalescing，所有row的部分都是由x变化引起， 而col的部分可以缓慢变化（或者由y引起）
    const uint innerRowB = threadIdx.x / BLOCK_N; // 这里用BLOCK_N是因为B是往K轴延伸，N是row的维度

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K)
    {
        // populate the SMEM caches
        As[innerRowA * BLOCK_K + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BLOCK_N + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BLOCK_K; // A是沿着横轴延伸，所以直接加
        B += BLOCK_K * N; // B是沿着纵轴延伸，所以需要乘以N

        // calculate per-thread results
        // 对shared memory的矩阵乘，所以先沿着K的维度走
        for (uint dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx)
        {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            // 每个线程，一个B算TM个A, 所以在这里把B存到寄存器里
            // 本来需要A沿着行走，B沿着列走，但是这里A在外层沿着行走的同时，内层不是一个元素了，而是直接一列A
            float tmpB = float(Bs[dotIdx * BLOCK_N + threadCol]); //由于矩阵乘，B需要往列走，所以要乘stride + threadCol
            for (uint resIdx = 0; resIdx < TM; ++resIdx)
            {
                threadResults[resIdx] +=
                    float(As[(threadRow * TM + resIdx) * BLOCK_K + dotIdx]) * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx)
    {
        C[(threadRow * TM + resIdx) * N + threadCol] = T(threadResults[resIdx]);
           
    }
}

// 主机端调用函数
template <typename T>
void runSgemm1DBlocktiling(
    const T *d_A, const T *d_B, T *d_C,
    const int M, const int N, const int K)
{
    const uint BLOCK_M = 64;
    const uint BLOCK_N = 64;
    const uint BLOCK_K = 8;
    const uint TM = 8;

    dim3 gridDim(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));
    dim3 blockDim((BLOCK_M * BLOCK_N) / TM);
    sgemm1DBlocktiling<BLOCK_M, BLOCK_N, BLOCK_K, TM, T><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
