import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    windows = input.unfold(0, window_size, 1)
    subarray_sum = windows.sum(dim=1)
    output.copy_(subarray_sum.max())