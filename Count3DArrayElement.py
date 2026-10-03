import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, P: int):
    output.copy_((input == P).sum())