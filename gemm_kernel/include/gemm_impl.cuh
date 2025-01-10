#pragma once

#include "../kernels/cuda_naive_gemm.cu"
#include "../kernels/cuda_mem_coales_gemm.cu"
#include "../kernels/cuda_shm_gemm.cu"
#include "../kernels/cuda_shm_1d_tiling.cu"
#include "../kernels/cuda_shm_2d_tiling.cu"
#include "../kernels/cute_naive_gemm.cu"
#include "../kernels/cute_shm_gemm.cu"
// 显式实例化所有模板
// template void naive_gemm<float>(const float*, const float*, float*, int, int, int);
// template void mem_coalesced_gemm<float>(const float*, const float*, float*, int, int, int);
// template void shm_gemm<float>(const float*, const float*, float*, int, int, int);
// template void runSgemm1DBlocktiling<float>(const float*, const float*, float*, int, int, int);
// template void runSgemm2DBlocktiling<float>(const float*, const float*, float*, int, int, int);
// template void cute_naive_gemm<float>(const float*, const float*, float*, int, int, int);
// template void cute_gemm_shm<float>(const float*, const float*, float*, int, int, int);

template void naive_gemm<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int);
template void mem_coalesced_gemm<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int);
template void shm_gemm<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int); 
template void runSgemm1DBlocktiling<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int); 
template void runSgemm2DBlocktiling<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int); 
template void cute_naive_gemm<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int);
template void cute_gemm_shm<cute::bfloat16_t>(const cute::bfloat16_t*, const cute::bfloat16_t*, cute::bfloat16_t*, int, int, int);

template void naive_gemm<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void mem_coalesced_gemm<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void shm_gemm<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void runSgemm1DBlocktiling<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void runSgemm2DBlocktiling<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void cute_naive_gemm<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);
template void cute_gemm_shm<cute::half_t>(const cute::half_t*, const cute::half_t*, cute::half_t*, int, int, int);