import torch

def solve(data: torch.Tensor, N: int):
    data.copy_(torch.sort(data).values)