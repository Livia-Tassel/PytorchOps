import torch

def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    # [A[0], B[0], A[1], B[1], A[2], B[2], ...]
    # [start : end : step]
    output[0::2] = A
    output[1::2] = B