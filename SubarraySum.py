import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    output.copy_(input[S: E+1].sum())