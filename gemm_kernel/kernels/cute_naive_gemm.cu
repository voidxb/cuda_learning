#include <cuda.h>
#include <stdlib.h>
#include "util.cuh"
#include <cuda_bf16.h>

// #define PRINT_INFO
using namespace cute;

// 为bfloat16添加显式转换函数
// CUTE_HOST_DEVICE
// inline cutlass::half_t convert(__nv_bfloat16 const& x) {
//     return cutlass::half_t(float(x));
// }
// MMA 定义部分
namespace cute_naive_mma
{ 
    using mma_op = SM80_16x8x16_F32F16F16F32_TN; // T为行优先， N为列优先
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),   // the best shape for perf, use 2x2=4 warps = 128 threads
                                        make_layout(Shape<_1, _1, _1>{}))); // influence little to perf
}

template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, typename TiledMMA>
__global__ void cute_naive_gemm_kernel(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k)
{   
    // 对 ABC三个矩阵进行Tensor封装
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    //Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));   // NxK , row major (外面我手动转置)
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));   // KxN, row major （和之前的匹配）
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    // 定位block id
    int bX = blockIdx.x;
    int bY = blockIdx.y;

    // 根据block id 找分块定位，make_coord 会自己定位，不用自己寻细节
    // y是M维度，x是N维度
    Tensor globalA = local_tile(A, make_tile(Int<BLOCK_M>{}, Int<BLOCK_K>{}), make_coord(bY, _)); // (BLOCK_M, BLOCK_K, num_tile_k)
    // NxK 只要Tensor封装符合内存分布，这里按照规定写就行
    Tensor globalB = local_tile(B, make_tile(Int<BLOCK_N>{}, Int<BLOCK_K>{}), make_coord(bX, _));  // (BLOCK_N, BLOCK_K, num_tile_k)
    Tensor globalC = local_tile(C, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}), make_coord(bY, bX)); // (BLOCK_M, BLOCK_N)


    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // 在 global memory上划分 ABC 的 tile
    auto globalTileA = thr_mma.partition_A(globalA);      // (MMA, MMA_M, MMA_K, num_tile_k)
    auto globalTileB = thr_mma.partition_B(globalB);       // (MMA, MMA_N, MMA_K, num_tile_k)
    auto globalTileC = thr_mma.partition_C(globalC);       // (MMA, MMA_M, MMA_N)
    // 寄存器级别的划分，gA/gB需要指定K维度的分块，gC不需要
    auto localTileA = thr_mma.partition_fragment_A(globalA(_, _, 0)); // 取K维度的第一个分块
    auto localTileB = thr_mma.partition_fragment_B(globalB(_, _, 0)); // 取K维度的第一个分块
    auto localTileC = thr_mma.partition_fragment_C(globalC(_, _));    // 不需要K维度分块
    clear(localTileC);

    // 从 gA 找到 tileK 的个数
    int num_tile_k = size<2>(globalA);
#pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile)
    {
        // global memory to register
        // just use cute::copy, not tiled
        copy(globalTileA(_, _, _, itile), localTileA);
        copy(globalTileB(_, _, _, itile), localTileB);

        // warp level, use  tiled_mma
        gemm(tiled_mma, localTileC, localTileA, localTileB, localTileC);
    }
    // register to global memory
    copy(localTileC, globalTileC);
}

template <typename T>
void cute_naive_gemm(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k)
{
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    // print block size
    // printf("block size: %d\n", size(cute_naive_mma::MMA{}));
    dim3 grid(CEIL_DIV(n, BLOCK_N), CEIL_DIV(m, BLOCK_M));
    dim3 block(size(cute_naive_mma::MMA{}));

    cute_naive_gemm_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K, cute_naive_mma::MMA><<<grid, block>>>(Aptr, Bptr, Cptr, m, n, k);
}


