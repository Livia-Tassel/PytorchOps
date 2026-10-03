import torch

def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):
    rotated_Q = torch.cat(
        (-Q[:, D//2:], Q[:, :D//2]),
        dim=1
    )
    result = Q * cos + rotated_Q * sin
    output.copy_(result)