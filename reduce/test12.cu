#include "error.cuh"
#include <stdio.h>
#include<math.h>
#include<ctime>
#include<iostream>
#include<cooperative_groups.h>

using namespace cooperative_groups;

int* construct_host_arr(int N, int v) {
	int*  matrix = (int *)malloc(sizeof(int) * N);
	for (int i = 0; i < N; i ++)
		*(matrix + i) = v;
	//memset(matrix, v, sizeof(int) * N);
	return matrix;
}

int * construct_dev_arr(int N) {
	int * matrix;
	CHECK(cudaMalloc((void **)&matrix, sizeof(int) * N));
	return matrix;
}

void __global__ reduce_final(int *d_x, int *d_y, int N)
{
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        extern __shared__ int s_y[];
	int y = 0;
	const int stride = gridDim.x * blockDim.x;
	for (int n = bid * blockDim.x + tid; n < N; n += stride)
		y += d_x[n];
        s_y[tid] = y;
        __syncthreads();
        for (int off = blockDim.x >> 1; off >= 32; off >>= 1) {
                if (tid < off)
                        s_y[tid] += s_y[tid + off];
                __syncthreads();
        }

        y = s_y[tid];
	thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
        for (int off = g.size() >> 1; off > 0; off >>=1) {
                y += g.shfl_down(y, off);
        }

        if (tid == 0) {
		d_y[bid] = y;
        }
}

int main(void) {
	cudaError_t err;	
	const int TILE_DIM = 128;
	long N = 10000000;
	long M = sizeof(int) * N;
	const int grid_size = (N + TILE_DIM - 1)/TILE_DIM;
	const int block_size = TILE_DIM;
	const int SMEM = sizeof(int) * TILE_DIM;
	int * d_A, *d_B, *h_A, *h_B;
	h_A = construct_host_arr(N, 1);
	h_B = construct_host_arr(N, 0);
	d_A = construct_dev_arr(N);
	d_B = construct_dev_arr(N);
	CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, M, cudaMemcpyHostToDevice));
	reduce_final<<<grid_size, block_size, SMEM>>>(d_A, d_B, N);
	//reduce_final<<<1, 1024, sizeof(int) * 1024>>>(d_B, d_B, grid_size);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
    		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    		// 处理错误
	}
	CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
	printf("sum = %d\n", h_B[0]);
	cudaFree(d_B);
	cudaFree(d_A);
	free(h_B);
	free(h_A);
}
