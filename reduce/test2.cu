#include "error.cuh"
#include<stdio.h>
#include<math.h>

const double EPSION = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z, int N);
void check(const double *x, const int N);

int main(void) {
	const int N = 100000000;
	const int M = sizeof(double) * N;
	cudaEvent_t start, stop;
	float elapsed_time;
	double * h_x = (double *)malloc(M);
	double * h_y = (double *)malloc(M);
	double * h_z = (double *)malloc(M);

	for (int n = 0; n < N; n++)
		h_x[n] = a, h_y[n] = b;

	double *d_x, *d_y, *d_z;

	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
	cudaEventQuery(start);

	CHECK(cudaMalloc((void **)&d_x, M));
	CHECK(cudaMalloc((void **)&d_y, M));
	CHECK(cudaMalloc((void **)&d_z, M));
	CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));
	const int block_size = 128;
	const int grid_size = (N-1)/ block_size + 1;
	add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
	check(h_z, N);
	free(h_x);
	free(h_y);
	free(h_z);

	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));
	CHECK(cudaFree(d_z));

	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Time = %g ms.\n", elapsed_time);
	CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
	return 0;
}

void check(const double *z, const int N) {
	bool has_err = false;
	for (int n = 0; n < N; ++n) {
		if (fabs(z[n] - c) > EPSION)
			has_err = true;
	}
	printf("%s:\n", has_err? "Error": "Succ");
}

void __device__ dev_add(const double x, const double y, double &z)
{
	z = x+ y;
}

void __global__ add(const double *x, const double *y, double *z, int N)
{
	const int n = blockDim.x * blockIdx.x + threadIdx.x;
	if (n < N)
		dev_add(x[n], y[n], z[n]);
}
