import torch

def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    x = input.reshape(N, C)
    mean = x.mean(dim=0)
    var = ((x - mean) ** 2).mean(dim=0)

    x_hat = (x - mean) / torch.sqrt(var + eps)
    result = gamma * x_hat + beta
    output.copy_(result.reshape_as(output))