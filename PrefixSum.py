import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    pre_sum = 0
    # torch.cumsum(input, dim=0)
    for i in range(N):
        output[i] = input[i] + pre_sum
        pre_sum += input[i]