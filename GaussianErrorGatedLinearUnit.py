import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    half = N // 2
    # GEGLU()
    output.copy_(input[:half] * 0.5 * input[half:] * (1 + torch.erf(input[half:] / (2 ** 0.5)))) 