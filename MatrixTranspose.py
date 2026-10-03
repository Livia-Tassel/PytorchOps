import torch

def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    # X: [rows, cols]
    # Y: [cols, rows]
    output.copy_(input.transpose(0, 1))