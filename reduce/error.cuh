#pragma once
#include <stdio.h>

#define CHECK(call)                                                          \
	do                                                                   \
	{                                                                    \
		const cudaError_t error_code = call;                         \
		if (error_code != cudaSuccess) {                             \
			printf("CudaError:\n");                              \
			printf("     File:         %s\n", __FILE__);         \
			printf("     Line:         %d\n", __LINE__);         \
			printf("     Err Code:     %d\n", error_code);       \
			printf("     Err Text:     %s\n", cudaGetErrorString(error_code));         \
			exit(1);                                             \
		}                                                            \
	} while (0)
