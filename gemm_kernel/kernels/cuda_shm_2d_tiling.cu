#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include "util.cuh"




template <const uint BLOCK_M, const uint BLOCK_N, const uint BLOCK_K, const uint TM, const uint TN, typename T>
__global__ void __launch_bounds__((BLOCK_M * BLOCK_N) / (TM * TN), 1)
    sgemm2DBlocktiling(const T *__restrict__ A,
                                   const T *__restrict__ B,
                                   T *__restrict__ C,
                                   const int M, const int N, const int K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BLOCK_M * BLOCK_N;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BLOCK_N/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BLOCK_N / TN);
  const int threadRow = threadIdx.x / (BLOCK_N / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BLOCK_M * BLOCK_K];
  __shared__ float Bs[BLOCK_K * BLOCK_N];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BLOCK_M * K;
  B += cCol * BLOCK_N;
  C += cRow * BLOCK_M * N + cCol * BLOCK_N;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BLOCK_K;
  const uint innerColA = threadIdx.x % BLOCK_K;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BLOCK_K;
  const uint innerRowB = threadIdx.x / BLOCK_N;
  const uint innerColB = threadIdx.x % BLOCK_N;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BLOCK_N;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BLOCK_M; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BLOCK_K + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BLOCK_K; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BLOCK_N + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BLOCK_K;     // move BLOCK_K columns to right
    B += BLOCK_K * N; // move BLOCK_K rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BLOCK_K + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BLOCK_N + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = T(threadResults[resIdxM * TN + resIdxN]);
    }
  }
}


template <typename T>
void runSgemm2DBlocktiling(
    const T *d_A, const T *d_B, T *d_C,
    const int M, const int N, const int K)
{
    const uint BLOCK_M = 64;
    const uint BLOCK_N = 64;
    const uint BLOCK_K = 8;
    const uint TM = 8;
    const uint TN = 8;

    dim3 blockDim(BLOCK_N/TN * BLOCK_M/TM);  // 使用常量而不是直接的数字
    dim3 gridDim(
        CEIL_DIV(N, BLOCK_N),
        CEIL_DIV(M, BLOCK_M)
    );

    sgemm2DBlocktiling<BLOCK_M, BLOCK_N, BLOCK_K, TM, TN, T><<<gridDim, blockDim>>>(
        d_A, d_B, d_C,
        M, N, K
    );
}