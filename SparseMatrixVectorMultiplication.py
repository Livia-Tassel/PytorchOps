import torch

def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    torch.mv(A.view(M, N), x, out=y)