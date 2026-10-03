import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # [-2, 1, -3] → [-0.02, 1, -0.03]
    output.copy_(torch.where(input < 0, 0.01 * input, input))