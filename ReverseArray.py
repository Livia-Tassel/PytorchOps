import torch

def solve(input: torch.Tensor, N: int):
    # [1, 2, 3, 4] → [4, 3, 2, 1]
    input.copy_(torch.flip(input, dims=[0]))