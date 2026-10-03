import torch

def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int
):
    # [1, 2, 3, 4, 5]
    """
    [[1, 2, 3],
     [2, 3, 4],
     [3, 4, 5]]
    """
    windows = input.unfold(0, kernel_size, 1)
    output.copy_(torch.matmul(windows, kernel))