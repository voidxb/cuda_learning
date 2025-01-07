#include <stdio.h>
#include "error.cuh"

const unsigned WIDTH = 8;
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff;

void __global__ test_warp_primitives(void);

int main(void)
{
	test_warp_primitives<<<1, BLOCK_SIZE>>>();
	CHECK(cudaDeviceSynchronize());
	return 0;
}

void __global__ test_warp_primitives(void)
{
	int tid = threadIdx.x;
	int lane_id = tid % WIDTH;

	if (tid == 0) printf("threadIdx.x ");
	printf("%2d ", tid);
	if (tid == 0) printf("\n");	

	if (tid == 0) printf("lane_id: ");   
        printf("%2d ", lane_id);
        if (tid == 0) printf("\n");   

	unsigned mask1 = __ballot_sync(FULL_MASK, tid > 0);
	unsigned mask2 = __ballot_sync(FULL_MASK, tid == 0);	
	if (tid == 0) printf("FULL MASK = %x\n", FULL_MASK);
	if (tid == 1) printf("mask1 = %x\n", mask1);
	if (tid == 0) printf("mask2 = %x\n", mask2);

	int result = __all_sync(FULL_MASK, tid);
	if (tid == 0) printf("all_sync (FULL_MASK): %d\n", result);
	result = __all_sync(mask1, tid);
	if (tid == 0) printf("all_sync (mask1): %d\n", result);
	result = __all_sync(mask2, tid);
	if (tid == 0) printf("all_sync (mask2): %d\n", result);
	result = __any_sync(FULL_MASK, tid);
        if (tid == 0) printf("any_sync (FULL_MASK): %d\n", result);
        result = __any_sync(mask1, tid);
        if (tid == 0) printf("any_sync (mask1): %d\n", result);
        result = __any_sync(mask2, tid);
        if (tid == 0) printf("any_sync (mask2): %d\n", result);

	int value = __shfl_sync(FULL_MASK, tid, 1, WIDTH);
	if (tid == 0) printf("shfl: ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

	value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
        if (tid == 0) printf("shfl up: ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

	value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
        if (tid == 0) printf("shfl down: ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");

	value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
        if (tid == 0) printf("shfl xor: ");
        printf("%2d ", value);
        if (tid == 0) printf("\n");
}
