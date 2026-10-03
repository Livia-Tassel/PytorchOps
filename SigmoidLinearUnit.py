import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # x * torch.sigmoid(x)
    output.copy_(input / (1 + torch.exp(-input)))