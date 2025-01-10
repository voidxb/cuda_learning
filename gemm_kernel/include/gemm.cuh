#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// GEMM kernel 声明
template<typename T>
__global__ void uncoalesced_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
);

template<typename T>
__global__ void mem_coalesced_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
);

template<typename T, const uint BLOCK_SIZE>
__global__ void shm_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
);

template <const uint BLOCK_M, const uint BLOCK_N, const uint BLOCK_K, const uint TM, typename T>
__global__ void sgemm1DBlocktiling(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
);

template <const uint BLOCK_M, const uint BLOCK_N, const uint BLOCK_K, const uint TM, const uint TN, typename T>
__global__ void sgemm2DBlocktiling(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ C,
    const int M, const int N, const int K
);


template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void cute_naive_gemm_kernel( const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k);

template <typename Config>
__global__ void gemm_opt_shm(const void *Aptr, const void *Bptr, void *Cptr, int m, int n,int k);

// 主机端函数声明
template<typename T>
void uncoalesced_gemm(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
);

template<typename T>
void mem_coalesced_gemm(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
);

template<typename T>
void shm_gemm(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
);

template<typename T>
void runSgemm1DBlocktiling(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
);

template<typename T>
void runSgemm2DBlocktiling(
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
);

template <typename T>
void cute_naive_gemm( const T *Aptr, const T *Bptr, T *Cptr,int m, int n, int k);

template <typename T>
void cute_gemm_shm(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k);

// 可选：添加一些常量定义
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
