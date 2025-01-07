#include "error.cuh"
#include <stdio.h>

int main(int argc, char *argv[])
{
	int device_id = 0;
	if (argc > 1) device_id = atoi(argv[1]);
	CHECK(cudaSetDevice(device_id));
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, device_id));
	printf("Device id:                 %d\n", device_id);
	printf("Device name:               %s\n", prop.name);
	printf("Compute capability:        %d.%d\n", prop.major, prop.minor);
	printf("Amount of global memory:   %g GB\n", prop.totalGlobalMem / (1024 * 1024 * 1024.0));
	printf("Amount of const memory:    %g KB\n", prop.totalConstMem / (1024.0));
	printf("Max grid size:             %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Max block dim size:        %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Number of SMs:             %d\n", prop.multiProcessorCount);
	printf("Max amount of shared memory per block: %g KB\n", prop.sharedMemPerBlock/1024.0);
	printf("Max amount of shared memory per SM:    %g KB\n", prop.sharedMemPerMultiprocessor/1024.0);
	printf("Max number of regs per block:          %d K\n",    prop.regsPerBlock/1024);
	printf("Max number of regs per SM:             %d K\n",    prop.regsPerMultiprocessor/1024);
	printf("Max number of threads per block:       %d\n",    prop.maxThreadsPerBlock);
	printf("Max number of threads per SM:       %d = %d warps\n",    prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerMultiProcessor/32);
	return 0;
}
