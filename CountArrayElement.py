import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    output.copy_((input == K).sum())