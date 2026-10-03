import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.sum(input))