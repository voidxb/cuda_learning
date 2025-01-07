#include "error.cuh"
#include<stdio.h>
#include<math.h>
#include<ctime>
#include<iostream>

#ifdef USE_DP
	typedef double real;
	const real EPSION = 1.0e3;
#else
	typedef float real;
	const real EPSION = 1.0e3f;
#endif

const real x0 = 100.0;

void cpu_arithmetic(real *x, const real x0, const int N)
{
	for (int n = 0; n < N; ++n)
	{
		real x_tmp = x[n];
		while (sqrt(x_tmp) < x0) {
			++x_tmp;
		}
		x[n] = x_tmp;
	}
}

void __global__ gpu_arithmetic(real *d_x, const real x0, const int N)
{
	const int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N) {
		real x_tmp = d_x[n];
		while (sqrt(x_tmp) < x0) {
			++x_tmp;
		}
		d_x[n] = x_tmp;
	}
}

int main(void) {
	const int N = 1000000;
	const int M = sizeof(real) * N;
	cudaEvent_t start, stop;
	float elapsed_time;
	real * h_x = (real *)malloc(M);
	real *d_x;

	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
	cudaEventQuery(start);

	CHECK(cudaMalloc((void **)&d_x, M));
	const int block_size = 128;
	const int grid_size = (N-1)/ block_size + 1;
	clock_t start_clk = clock();
	//gpu_arithmetic<<<grid_size, block_size>>>(d_x, x0, N);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(h_x, d_x, M, cudaMemcpyDeviceToHost));
	cpu_arithmetic(h_x, x0, N);
	clock_t end_clk = clock();
	// 计算持续时间（以秒为单位）
	double elapsed_clk = double(end_clk - start_clk) / CLOCKS_PER_SEC;
	// 输出结果
	std::cout << "Elapsed time: " << elapsed_clk << " seconds" << std::endl;

	free(h_x);
	CHECK(cudaFree(d_x));

	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Time = %g ms.\n", elapsed_time);
	CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
	return 0;
}
