#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemm_impl.cuh"
#include "benchmark.cuh"
#include "util.cuh"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>\n";
        return 1;
    }

    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);

    using compute_t = cute::half_t; // 或者使用 float
    save_gemm_config(M, N, K, get_compute_type_str<compute_t>(), "results/gemm_config.json");
    // using compute_t = float;
    // using compute_t = cute::bfloat16_t;
    // 分配并初始化输入矩阵
    compute_t *h_A = new compute_t[M * K];
    compute_t *h_B = new compute_t[K * N];

    // 初始化输入数据
    fill_randn(h_A, M * K);
    fill_randn(h_B, K * N);

    // 分配设备内存
    compute_t *d_A, *d_B;
    cudaMalloc(&d_A, M * K * sizeof(compute_t));
    cudaMalloc(&d_B, K * N * sizeof(compute_t));

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(compute_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(compute_t), cudaMemcpyHostToDevice);

    // 运行所有GEMM实现的基准测试
    benchmark_all_gemm<compute_t>(d_A, d_B, M, N, K, 20, true);

    // 清理内存
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;

    return 0;
}