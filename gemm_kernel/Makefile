# 使用第一个GPU的compute capability，并确保格式正确
CUDA_ARCH=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n 1)
CC_FILES=$(shell find ./ -maxdepth 1 -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)


all:$(EXE_FILES)

%:%.cu
	nvcc -o $@_exe $< -O2 -arch=sm_$(CUDA_ARCH) -std=c++17 -Ithird_party/cutlass/include -Iinclude --expt-relaxed-constexpr -cudart shared --cudadevrt none 

clean:
	rm -rf *_exe

main:
	nvcc -o main_exe main.cu -O2 -arch=sm_$(CUDA_ARCH) -std=c++17 -Ithird_party/cutlass/include -Iinclude --expt-relaxed-constexpr -cudart shared --cudadevrt none 

main360:
	nvcc -o main_exe main.cu -O2 -arch=sm_$(CUDA_ARCH) -std=c++17 -Idevelop/cutlass/include -Iinclude --expt-relaxed-constexpr -cudart shared --cudadevrt none 