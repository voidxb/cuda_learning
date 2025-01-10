#include <cuda_bf16.h>
#include <cuda_runtime.h>


template<typename T, const uint BLOCK_SIZE>
__global__ void shm_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M, const int N, const int K
) {
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // 计算当前线程负责的输出元素位置(根据block idx)
    // 因为是row major, 所有把threadIdx.x 作为col_idx为了memory coalescing(内存合并)
    const int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    // 累加器，用于存储计算结果
    float sum = 0.0f;
    
    // 遍历所有需要的分块
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // 协作加载数据到共享内存
        if (row_idx < M && tile * BLOCK_SIZE + threadIdx.x < K) {
            // 矩阵A是 M * K, 取tile是沿着K方向（row）, 所有用threadIdx.x来作为连续index
            // row_idx * K, 是起始位置, K是stride, row_idx由threadIdx.y决定纵轴
            // tile * BLOCK_SIZE是当前的tile起始位置
            // threadIdx.x 提供当前tile的连续取值
            As[threadIdx.y][threadIdx.x] = A[row_idx * K + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = T(0.0f);
        }
        
        if (col_idx < N && tile * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col_idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = T(0.0f);
        }
        
        // 同步以确保所有数据都已加载完成
        __syncthreads();
        
        // 计算当前分块的部分和
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += float(As[threadIdx.y][k]) * float(Bs[k][threadIdx.x]);
        }
        // 同步以确保计算完成后再加载下一个分块
        __syncthreads();
    }
    
    // 写回结果
    if (row_idx < M && col_idx < N) {
        C[row_idx * N + col_idx] = T(sum);
    }
}

// 主机端调用函数
template<typename T>
void shm_gemm(  // 返回值改为 float，用于返回执行时间
    const T* d_A, const T* d_B, T* d_C,
    const int M, const int N, const int K
) {
    const uint BLOCK_SIZE = 32;
    dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 numBlocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    // 启动kernel
    shm_gemm_kernel<T, BLOCK_SIZE><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    
}