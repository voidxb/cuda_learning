#include "error.cuh"
#include <stdio.h>
#include<math.h>
#include<ctime>
#include<iostream>

const int TILE_DIM = 32;

int* construct_host_matrix(int N, int v) {
	int*  matrix = (int *)malloc(sizeof(int) * N * N);
	memset(matrix, v, sizeof(int) * N * N);
	return matrix;
}

int * construct_dev_matrix(int N) {
	int * matrix;
	CHECK(cudaMalloc((void **)&matrix, sizeof(int) * N * N));
	return matrix;
}

__global__ void matrix_transpose1(const int *A, int *B, const int N)
{
	const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
	const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
	if (nx < N && ny < N)
		B[nx * N + ny] = A[ny * N + nx];
}

__global__ void matrix_transpose2(const int *A, int *B, const int N)
{
        const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
        const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
        if (nx < N && ny < N)
                B[ny * N + nx] = A[nx * N + ny];
}

__global__ void matrix_transpose3(const int *A, int *B, const int N)
{
        const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
        const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
        if (nx < N && ny < N)
                B[ny * N + nx] = __ldg(&A[nx * N + ny]);
}

__global__ void matrix_transpose4(const int *A, int *B, const int N)
{
	__shared__ int S[TILE_DIM][TILE_DIM + 1];
	int bx = blockIdx.x * TILE_DIM;
	int by = blockIdx.y * TILE_DIM;
	
	int nx1 = bx + threadIdx.x;
	int ny1 = by + threadIdx.y;
	if (nx1 < N && ny1 < N)
		S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
	__syncthreads();
	int nx2 = bx + threadIdx.y;
        int ny2 = by + threadIdx.x;
        if (nx2 < N && ny2 < N)
                B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
}

int main(void) { 
	const int TILE_DIM = 32;
	int N = 10000;
        int M = sizeof(int ) * N * N;
	const int grid_size_x = (N + TILE_DIM - 1)/TILE_DIM;
	const int grid_size_y = grid_size_x;
	const dim3 block_size(TILE_DIM, TILE_DIM);
	const dim3 grid_size(grid_size_x, grid_size_y);
	int * d_A, *d_B, *h_A, *h_B;
	h_A = construct_host_matrix(N, 1);
	h_B = construct_host_matrix(N, 0);
	d_A = construct_dev_matrix(N);
	d_B = construct_dev_matrix(N);
	CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));
	matrix_transpose4<<<grid_size, block_size>>>((const int *)d_A, d_B, N);
	CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
	for (int i = 0; i < N; i ++)
		for (int j = 0; j < N; j++)
			if (h_A[i * N + j] != h_B[j * N + i])
				printf("A != B, idx(%d, %d), val(%d, %d)\n", i, j, h_A[i * N + j], h_B[i * N + j]);
}
