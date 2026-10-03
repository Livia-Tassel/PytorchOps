import torch

def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    C.copy_(alpha * (A.view(M, K) @ B.view(K, N)) + beta * C.view(M, N))