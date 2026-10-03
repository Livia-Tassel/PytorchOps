import torch

def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    result.copy_((A * B).sum().reshape_as(result))