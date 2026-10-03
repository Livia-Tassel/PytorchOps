import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # [-2, 1, -3] → [0, 1, 0]
    torch.clamp(input, min=0, out=output)