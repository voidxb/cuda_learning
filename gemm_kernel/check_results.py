import json
import torch
import numpy as np

with open("results/gemm_config.json", "r") as f:
    config = json.load(f)

M = config["matrix_dims"]["M"]
N = config["matrix_dims"]["N"]
K = config["matrix_dims"]["K"]
compute_type = config["compute_type"]

dtype_map = {
    "half_t": torch.half,
    "float": torch.float,
    "bfloat16_t": torch.bfloat16
}

dtype = dtype_map[compute_type]

def load_bfloat16_bin(filename, shape):
    # 读取二进制文件
    data = np.fromfile(filename, dtype=np.uint16)
    # 转换为 torch.bfloat16
    return torch.frombuffer(data.tobytes(), dtype=dtype).reshape(shape).cuda().float()

# 设置维度
# M, N, K = 65536, 128, 1024

# 读取矩阵
A = load_bfloat16_bin("results/matrix_A.bin", (M, K))
B = load_bfloat16_bin("results/matrix_B.bin", (K, N))
B_transpose = load_bfloat16_bin("results/matrix_B_transpose.bin", (N, K))
C_naive = load_bfloat16_bin("results/matrix_C_naive.bin", (M, N))
C_coalesced = load_bfloat16_bin("results/matrix_C_coalesced.bin", (M, N))
C_shared = load_bfloat16_bin("results/matrix_C_shared.bin", (M, N))
C_shared_1d_tiling = load_bfloat16_bin("results/matrix_C_shared_1d_tiling.bin", (M, N))
C_shared_2d_tiling = load_bfloat16_bin("results/matrix_C_shared_2d_tiling.bin", (M, N))
C_cute_naive = load_bfloat16_bin("results/matrix_C_cute_naive.bin", (M, N))
C_cute_shm = load_bfloat16_bin("results/matrix_C_cute_shm.bin", (M, N))

# 计算参考结果
C_ref = torch.mm(A, B)
print("C_ref computed")
cosine_naive = torch.nn.functional.cosine_similarity(C_naive.view(-1), C_ref.view(-1), dim=-1)
cosine_coalesced = torch.nn.functional.cosine_similarity(C_coalesced.view(-1), C_ref.view(-1), dim=-1)
cosine_shared = torch.nn.functional.cosine_similarity(C_shared.view(-1), C_ref.view(-1), dim=-1)
cosine_shared_1d_tiling = torch.nn.functional.cosine_similarity(C_shared_1d_tiling.view(-1), C_ref.view(-1), dim=-1)
cosine_shared_2d_tiling = torch.nn.functional.cosine_similarity(C_shared_2d_tiling.view(-1), C_ref.view(-1), dim=-1)
cosine_cute_naive = torch.nn.functional.cosine_similarity(C_cute_naive.view(-1), C_ref.view(-1), dim=-1)
cosine_cute_shm = torch.nn.functional.cosine_similarity(C_cute_shm.view(-1), C_ref.view(-1), dim=-1)
print(f"naive cosine_similarity: {cosine_naive}")
print(f"coalesced cosine_similarity: {cosine_coalesced}")
print(f"shared cosine_similarity: {cosine_shared}")
print(f"shared_1d_tiling cosine_similarity: {cosine_shared_1d_tiling}")
print(f"shared 2d tiling cosine_similarity: {cosine_shared_2d_tiling}")
print(f"cute_naive cosine_similarity: {cosine_cute_naive}")
print(f"cute_shm cosine_similarity: {cosine_cute_shm}")
# print elemets

# print("ref")
# print(C_ref[:3, :3])
# print("naive")
# print(C_naive[:3, :3])
# print("coalesced")
# print(C_coalesced[:3, :3])
# print("shared")
# print(C_shared[:3, :3])
# print("shared_1d_tiling")
# print(C_shared_1d_tiling[:3, :3])
# print("shared_2d_tiling")
# print(C_shared_2d_tiling[:3, :3])
# print("cute_naive")
# print(C_cute_naive[:3, :3])
# print("cute_shm")
# print(C_cute_shm[:3, :3])
