import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    half = N // 2
    output.copy_(input[:half] * torch.sigmoid(input[:half]) * input[half:])