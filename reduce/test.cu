#include <stdio.h>

__global__ void hello_from_gpu()
{
	const int bid = blockIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	printf("Hello world from GPU! from block %d, thread(%d, %d)\n", bid, tidx, tidy);
}

int main(void) {
	const dim3 block_size(2, 4);
	hello_from_gpu<<<2, block_size>>>();
	cudaDeviceSynchronize();
	return 0;
}
