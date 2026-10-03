import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    # torch,topk(x, k)
    values = input.sort(descending=True).values
    output.copy_(values[:k])