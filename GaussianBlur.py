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

    pad_h = kernel_rows // 2
    pad_w = kernel_cols // 2
    input = torch.nn.functional.pad(
        input,
        (pad_w, pad_w, pad_h, pad_h)
    )
    
    windows = (
        input.unfold(0, kernel_rows, 1)
             .unfold(1, kernel_cols, 1)
    )
    # [2, 2, 2, 2] * [2, 2]
    result = (windows * kernel).sum(dim=(-1, -2))
    result = result.reshape(-1)
    output.copy_(result) 