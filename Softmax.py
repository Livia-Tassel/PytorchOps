import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    # torch.softmax(x)
    max_val = torch.max(input)
    exp_x = torch.exp(input - max_val)
    output.copy_(exp_x / torch.sum(exp_x))