#pragma once

#include "util.cuh"
#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "gemm.cuh"

// 在benchmark_all_gemm模板函数之前添加这些声明
template<typename T>
void write_matrix_to_file(const char* filename, const T* data, size_t size) {
    ::write_to_file<T>(filename, data, size);
}

// 添加一个辅助函数来执行单个GEMM实现的benchmark
template<typename T>
float benchmark_single_gemm(
    auto& gemm_func,  // 使用auto&来接收函数引用
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K,
    int repeat_times
) {
    float total_time = 0.0f;
    for (int i = 0; i < repeat_times; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        gemm_func(d_A, d_B, d_C, M, N, K);  // 调用传入的GEMM实现
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return total_time / repeat_times;
}

// 创建一个统一的接口来运行所有GEMM实现
template<typename T>
void benchmark_all_gemm(
    const T* d_A, const T* d_B,
    const int M, const int N, const int K,
    int repeat_times = 100,
    bool write_to_file = false
) {
    // 为每种实现分配设备内存
    T *d_C_uncoalesced, *d_C_coalesced, *d_C_shared, *d_C_shared_1d_tiling, *d_C_shared_2d_tiling, *d_C_cute_naive, *d_C_cute_shm, *d_B_transpose;
    cudaMalloc(&d_C_uncoalesced, M * N * sizeof(T));
    cudaMalloc(&d_C_coalesced, M * N * sizeof(T));
    cudaMalloc(&d_C_shared, M * N * sizeof(T));
    cudaMalloc(&d_C_shared_1d_tiling, M * N * sizeof(T));
    cudaMalloc(&d_C_shared_2d_tiling, M * N * sizeof(T));
    cudaMalloc(&d_C_cute_naive, M * N * sizeof(T));
    cudaMalloc(&d_C_cute_shm, M * N * sizeof(T));
    cudaMalloc(&d_B_transpose, K * N * sizeof(T));

    cudaMemcpy(d_B_transpose, d_B, K * N * sizeof(T), cudaMemcpyDeviceToDevice);
    transpose_matrix<T>(d_B_transpose, K, N);

    
    printf("\n=== GEMM Performance Benchmark ===\n");
    
    // 使用新的benchmark函数测试各种实现
    float avg_time = benchmark_single_gemm<T>(naive_gemm<T>, d_A, d_B, d_C_uncoalesced, M, N, K, repeat_times);
    printf("未优化GEMM平均执行时间: %.3f ms\n", avg_time);

    avg_time = benchmark_single_gemm<T>(mem_coalesced_gemm<T>, d_A, d_B, d_C_coalesced, M, N, K, repeat_times);
    printf("内存合并GEMM平均执行时间: %.3f ms\n", avg_time);

    avg_time = benchmark_single_gemm<T>(shm_gemm<T>, d_A, d_B, d_C_shared, M, N, K, repeat_times);
    printf("共享内存GEMM平均执行时间: %.3f ms\n", avg_time);

    avg_time = benchmark_single_gemm<T>(runSgemm1DBlocktiling<T>, d_A, d_B, d_C_shared_1d_tiling, M, N, K, repeat_times);
    printf("共享内存1D分块GEMM平均执行时间: %.3f ms\n", avg_time);

    avg_time = benchmark_single_gemm<T>(runSgemm2DBlocktiling<T>, d_A, d_B, d_C_shared_2d_tiling, M, N, K, repeat_times);
    printf("共享内存2D分块GEMM平均执行时间: %.3f ms\n", avg_time);

    // 对于cute实现，需要特殊处理
    using cute_T = typename std::conditional<std::is_same<T, __nv_bfloat16>::value,
                                           cute::bfloat16_t,
                                           T>::type;
    
    avg_time = benchmark_single_gemm<T>(cute_naive_gemm<cute_T>, d_A, d_B_transpose, d_C_cute_naive, M, N, K, repeat_times);
    printf("cute_naive_gemm平均执行时间: %.3f ms\n", avg_time);

    avg_time = benchmark_single_gemm<T>(cute_gemm_shm<T>, d_A, d_B_transpose, d_C_cute_shm, M, N, K, repeat_times);
    printf("cute_gemm_shm平均执行时间: %.3f ms\n", avg_time);

    if (write_to_file) {
        write_matrix_to_file("results/matrix_A.bin", d_A, M * K);
        write_matrix_to_file("results/matrix_B.bin", d_B, K * N);
        write_matrix_to_file("results/matrix_C_naive.bin", d_C_uncoalesced, M * N);
        write_matrix_to_file("results/matrix_C_coalesced.bin", d_C_coalesced, M * N);
        write_matrix_to_file("results/matrix_C_shared.bin", d_C_shared, M * N);
        write_matrix_to_file("results/matrix_C_shared_1d_tiling.bin", d_C_shared_1d_tiling, M * N);
        write_matrix_to_file("results/matrix_C_shared_2d_tiling.bin", d_C_shared_2d_tiling, M * N);
        write_matrix_to_file("results/matrix_C_cute_naive.bin", d_C_cute_naive, M * N);
        write_matrix_to_file("results/matrix_B_transpose.bin", d_B_transpose, K * N);
        write_matrix_to_file("results/matrix_C_cute_shm.bin", d_C_cute_shm, M * N);
    }
    // 释放临时内存
    cudaFree(d_C_uncoalesced);
    cudaFree(d_C_coalesced);
    cudaFree(d_C_shared);
    cudaFree(d_C_shared_1d_tiling);
    cudaFree(d_C_shared_2d_tiling);
    cudaFree(d_C_cute_naive);
    cudaFree(d_C_cute_shm);
    cudaFree(d_B_transpose);
    printf("==============================\n\n");
} 