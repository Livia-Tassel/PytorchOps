import torch

def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    x = input
    rms = torch.sqrt((x * x).mean() + eps)
    x_hat = x / rms
    output.copy_(gamma * x_hat + beta)