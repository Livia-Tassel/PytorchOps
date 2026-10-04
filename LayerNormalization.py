import torch

def solve(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    mean = input.mean(dim=1, keepdim=True)
    var = ((input - mean) ** 2).mean(
        dim=1,
        keepdim=True
    )
    normalized = (input - mean) / torch.sqrt(var + eps)
    result = normalized * weight + bias
    output.copy_(result)