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

void __global__ reduce1(int *d_x, int *d_y, int N)
{
	const int tid = threadIdx.x;
	int *x = d_x + blockDim.x * blockIdx.x;
	for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
		if ((tid < off) && (blockDim.x * blockIdx.x + tid < N))
			x[tid] += x[tid + off];
		__syncthreads();
	}

	if (tid == 0) {
		d_y[blockIdx.x] = x[0];
	}
}

void __global__ reduce2(int *d_x, int *d_y, int N)
{
        const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int n = bid * blockDim.x + tid;
	__shared__ int s_y[128];
	s_y[tid] = (n < N)?d_x[n]:0;
	__syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if ((tid < off) && (n < N))
                        s_y[tid] += s_y[tid + off];
                __syncthreads();
        }

        if (tid == 0) {
                d_y[bid] = s_y[0];
        }
}

void __global__ reduce3(int *d_x, int *d_y, int N)
{
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int n = bid * blockDim.x + tid;
        __shared__ int s_y[128];
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

int main(void) {
	cudaError_t err;	
	const int TILE_DIM = 128;
	long N = 10000000;
	long M = sizeof(int) * N;
	const int grid_size = (N + TILE_DIM - 1)/TILE_DIM;
	const int block_size = TILE_DIM;
	int * d_A, *d_B, *h_A, *h_B;
	int sum = 0;
	h_A = construct_host_arr(N, 1);
	h_B = construct_host_arr(N, 0);
	d_A = construct_dev_arr(N);
	d_B = construct_dev_arr(N);
	CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, M, cudaMemcpyHostToDevice));
	reduce3<<<grid_size, block_size>>>(d_A, d_B, N);
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
