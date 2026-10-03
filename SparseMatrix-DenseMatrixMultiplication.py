import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, nnz: int):
    A = A.view(M, N)
    B = B.view(N, K)
    C.copy_(A @ B)