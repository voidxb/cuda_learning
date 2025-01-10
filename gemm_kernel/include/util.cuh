#pragma once

#include <cuda_runtime.h>
#include <random>
#include <cute/tensor.hpp>


#include <fstream>
#include <string>

// 添加新的配置保存函数
void save_gemm_config(int M, int N, int K, const std::string& compute_type, const std::string& output_path) {
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", output_path.c_str());
        return;
    }
    
    output_file << "{\n";
    output_file << "    \"matrix_dims\": {\n";
    output_file << "        \"M\": " << M << ",\n";
    output_file << "        \"N\": " << N << ",\n";
    output_file << "        \"K\": " << K << "\n";
    output_file << "    },\n";
    output_file << "    \"compute_type\": \"" << compute_type << "\"\n";
    output_file << "}\n";
    
    output_file.close();
}

// 获取计算类型的字符串表示
template<typename T>
std::string get_compute_type_str() {
    if (std::is_same<T, cute::half_t>::value) return "half_t";
    if (std::is_same<T, float>::value) return "float";
    if (std::is_same<T, cute::bfloat16_t>::value) return "bfloat16_t";
    return "unknown";
}

#ifndef PRINT
#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");
#endif

#ifndef PRINTTENSOR
#define PRINTTENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");
#endif
// 添加CEIL_DIV宏定义
#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return milliseconds;
    }
};

template<typename T>
void fill_randn(T* data, size_t size, float mean = 0.0f, float stddev = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, stddev);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = T(d(gen));
    }
}

// 显式实例化
template void fill_randn<float>(float*, size_t, float, float);
template void fill_randn<__nv_bfloat16>(__nv_bfloat16*, size_t, float, float); 
template void fill_randn<cute::half_t>(cute::half_t*, size_t, float, float); 
template<typename T>
void write_to_file(const char* filename, const T* data, size_t size) {
    T* h_data = new T[size];
    cudaMemcpy(h_data, data, size * sizeof(T), cudaMemcpyDeviceToHost);
    FILE *fp = fopen(filename, "wb");
    fwrite(h_data, sizeof(T), size, fp);
    fclose(fp);
    delete[] h_data;
}

// 显式实例化 write_to_file 函数
template void write_to_file<float>(const char*, const float*, size_t);
template void write_to_file<__nv_bfloat16>(const char*, const __nv_bfloat16*, size_t);
template void write_to_file<cute::half_t>(const char*, const cute::half_t*, size_t);
// 添加转置kernel
template<typename T>
__global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// 添加转置函数封装
template<typename T>
void transpose_matrix(T* d_input, int rows, int cols) {
    T* d_temp;
    cudaMalloc(&d_temp, rows * cols * sizeof(T));
    
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, 
              (rows + block.y - 1) / block.y);
    
    transpose_kernel<<<grid, block>>>(d_input, d_temp, rows, cols);
    
    // 将结果复制回原始内存
    cudaMemcpy(d_input, d_temp, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_temp);
}

// 显式实例化
template void transpose_matrix<float>(float*, int, int);
template void transpose_matrix<__nv_bfloat16>(__nv_bfloat16*, int, int);
template void transpose_matrix<cute::half_t>(cute::half_t*, int, int);
