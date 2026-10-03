import torch

def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    # counts = torch.bincount(input.long(), minlength=num_bins)
    # histogram.copy_(counts[:num_bins].to(histogram.dtype))
    for i in range(num_bins):
        histogram[i] = (input == i).sum()