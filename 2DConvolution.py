import torch

def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input = input.view(-1, input_cols)
    kernel = kernel.view(-1, kernel_cols)
    windows = (
        input.unfold(0, kernel_rows, 1)
             .unfold(1, kernel_cols, 1)
    )
    # [2, 2, 2, 2] * [2, 2]
    result = (windows * kernel).sum(dim=(-1, -2))
    result = result.reshape(-1)
    output.copy_(result)