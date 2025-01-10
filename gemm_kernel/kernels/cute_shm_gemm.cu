#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "util.cuh"

namespace config
{
    using namespace cute;
    template <typename T_, int BLOCK_M_ = 128, int BLOCK_N_ = 128, int BLOCK_K_ = 32>
    struct GemmConfigV1
    {
        using T = T_;

        static constexpr int BLOCK_M = BLOCK_M_;
        static constexpr int BLOCK_N = BLOCK_N_;
        static constexpr int BLOCK_K = BLOCK_K_;

        // shared memory layout
        using SmemLayoutAtom = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<8>{}, Int<BLOCK_K>{}),
                        make_stride(Int<BLOCK_K>{}, Int<1>{}))));
        using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                                   make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{})));
        using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                                   make_shape(Int<BLOCK_N>{}, Int<BLOCK_K>{})));
        // mma
        using mma_op = SM80_16x8x16_F32F16F16F32_TN;
        using mma_traits = MMA_Traits<mma_op>;
        using mma_atom = MMA_Atom<mma_traits>;
        using MMA = decltype(make_tiled_mma(mma_atom{},
                                            make_layout(Shape<_2, _2, _1>{}),
                                            make_layout(Shape<_2, _2, _1>{})));
        // copy from global memory to shared memory
        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
        using G2SCopyA =
            decltype(make_tiled_copy(g2s_copy_atom{},
                                     make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                 make_stride(Int<4>{}, Int<1>{})),
                                     make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using G2SCopyB = G2SCopyA;

        // copy from shared memory to register
        // use mma tiled ,so no tiled here
        using s2r_copy_op = SM75_U32x4_LDSM_N;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
        using S2RCopyAtomA = s2r_copy_atom;
        using S2RCopyAtomB = s2r_copy_atom;

        // C_shm is shared with A_shm and B_shm
        static constexpr int shm_size_AB =
            cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
        static constexpr int kShmSize =
            shm_size_AB * sizeof(T);
    };
}

template <typename Config>
__global__ void
gemm_opt_shm(const void *Aptr, const void *Bptr, void *Dptr, int m, int n,
             int k)
{
    using T = typename Config::T;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using TiledMMA = typename Config::MMA;
    // using TiledMMA = cute_naive_mma_shm::MMA;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;

    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int BLOCK_K = Config::BLOCK_K;

    // extern __shared__ T shm_data[];

    extern __shared__ char shm_data_char[];
    T *shm_data = reinterpret_cast<T *>(shm_data_char);

    T *Ashm = shm_data;
    T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                           make_stride(k, Int<1>{})); // (M, K)
    Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                           make_stride(k, Int<1>{})); // (N, K)
    Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                           make_stride(n, Int<1>{})); // (M, N)
    // global memory
    Tensor gA = local_tile(A, make_tile(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
                           make_coord(iy, _)); // (kTileM, kTileK, k)
    Tensor gB = local_tile(B, make_tile(Int<BLOCK_N>{}, Int<BLOCK_K>{}),
                           make_coord(ix, _)); // (kTileN, kTileK, k)
    Tensor gD = local_tile(D, make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                           make_coord(iy, ix)); // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
                          SmemLayoutA{}); // (kTileM, kTileK)
    auto sB = make_tensor(make_smem_ptr(Bshm),
                          SmemLayoutB{}); // (kTileN, kTileK)

    // register, use tiled_mma to partition register A/B/C
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

    auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)
    // fill zero for accumulator
    clear(tCrD);

    // from global memory to shared memory
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy =
        g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy =
        g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)
    using TiledMMAType = typename decltype(tiled_mma)::TiledMMA;
    // from shared memory to register, use tiled_mma to generate tiled_copy
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    // loop over k: i. load tile, ii. mma

    // if (idx == 0 && ix == 0 && iy == 0)
    // {
    //     printf("Source shape: %d x %d\n", 
    //     int(size<0>(tAsA(_, _, 0))), 
    //     int(size<1>(tAsA(_, _, 0))));
    //     printf("Destination shape: %d x %d\n", 
    //     int(size<0>(tCrA_view(_, _, 0))), 
    //     int(size<1>(tCrA_view(_, _, 0))));
    //     PRINT("MMA thread layout",thr_mma);
    //     PRINT("S2R Copy A layout", s2r_tiled_copy_a);
    //     PRINT("tAsA layout", tAsA);
    //     PRINT("tCrA_view layout", tCrA_view);
    // }
    int ntile = k / BLOCK_K;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile)
    {
        // copy  (CPY, CPY_M, CPY_K) , async
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile),
                   tAsA_copy(_, _, _));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile),
                   tBsB_copy(_, _, _));
        cp_async_fence();

        cp_async_wait<0>();
        __syncthreads();

        int nk = size<2>(tCrA);
    

#pragma unroll
        for (int ik = 0; ik < nk; ++ik)
        {
            // copy  (CPY, CPY_M), sync
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik),
                       tCrA_view(_, _, ik));
            // copy  (CPY, CPY_N)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik),
                       tCrB_view(_, _, ik));
            // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik
    } // itile

    // register to global memory
    cute::copy(tCrD, tCgD);
}

template <typename T>
void cute_gemm_shm(const T *Aptr, const T *Bptr, T *Cptr, int m, int n, int k)
{
    using namespace cute;
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;

    // 实例化gemm_config
    config::GemmConfigV1<T, BLOCK_M, BLOCK_N, BLOCK_K> gemm_config;
    dim3 grid(CEIL_DIV(n, BLOCK_N), CEIL_DIV(m, BLOCK_M));

    // 从gemm_config里获取线程数
    size_t block_size = size(typename decltype(gemm_config)::MMA{});
    dim3 block(block_size);

    // 这里的shm_size是从SmemLayoutA和SmemLayoutB计算得来的
    int shm_size = gemm_config.kShmSize;

    // 限定动态共享内存的大小
    cudaFuncSetAttribute(gemm_opt_shm<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    gemm_opt_shm<decltype(gemm_config)>
        <<<grid, block, shm_size>>>(Aptr, Bptr, Cptr, m, n, k);
}
