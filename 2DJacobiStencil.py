import torch

def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    output.copy_(input)
    output[1:-1, 1:-1] = (
        input[:-2, 1:-1] +
        input[2:, 1:-1] +
        input[1:-1, :-2] +
        input[1:-1, 2:]
    ) / 4