import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    # A: [M, N]
    # B: [N, K]
    # C = A @ B → [M, K]
    torch.matmul(A, B, out=C)