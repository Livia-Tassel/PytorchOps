import torch

def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    output.copy_(torch.clamp(input, min=lo, max=hi))