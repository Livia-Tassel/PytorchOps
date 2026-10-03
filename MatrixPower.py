import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int, P: int):
    result = torch.linalg.matrix_power(input.view(N, N), P)
    output.copy_(result.reshape_as(output))