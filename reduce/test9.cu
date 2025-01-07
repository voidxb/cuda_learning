#include "error.cuh"
#include <stdio.h>
#include<math.h>
#include<ctime>
#include<iostream>

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

void __global__ reduce(int *d_x, int *d_y, int N)
{
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid * blockDim.x + tid;
        extern __shared__ int s_y[];
        s_y[tid] = (n < N)?d_x[n]:0;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if ((tid < off) && (n < N))
                        s_y[tid] += s_y[tid + off];
                __syncthreads();
        }

        if (tid == 0) {
                atomicAdd(&d_y[0], s_y[0]);
        }
}

void __global__ reduce_warp(int *d_x, int *d_y, int N)
{
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid * blockDim.x + tid;
        extern __shared__ int s_y[];
        s_y[tid] = (n < N)?d_x[n]:0;
	__syncthreads();
        for (int off = blockDim.x >> 1; off >= 32; off >>= 1) {
                if ((tid < off) && (n < N))
                        s_y[tid] += s_y[tid + off];
                __syncthreads();
        }

	for (int off = 16; off > 0; off >>=1) {
		if (tid < off)
			s_y[tid] += s_y[tid + off];
		__syncwarp();	
	}

        if (tid == 0) {
                atomicAdd(&d_y[0], s_y[0]);
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
	int sum = 0;
	h_A = construct_host_arr(N, 1);
	h_B = construct_host_arr(N, 0);
	d_A = construct_dev_arr(N);
	d_B = construct_dev_arr(N);
	CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, M, cudaMemcpyHostToDevice));
	reduce_warp<<<grid_size, block_size, SMEM>>>(d_A, d_B, N);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
    		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    		// 处理错误
	}
	CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i ++) {
		if (*(h_B + i) != 0) {
			printf("b[%d]= %d \n", i, h_B[i]);
			sum += h_B[i];
		}
	}
	printf("sum = %d\n", sum);
}
