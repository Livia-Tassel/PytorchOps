import torch

def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input = input.view(-1, input_rows, input_cols)
    kernel = kernel.view(-1, kernel_rows, kernel_cols)
    windows = (
        input.unfold(0, kernel_depth, 1)
             .unfold(1, kernel_rows, 1)
             .unfold(2, kernel_cols, 1)
    )

    result = (windows * kernel).sum(dim=(-1, -2, -3))
    result = result.reshape(-1)
    output.view(-1).copy_(result)