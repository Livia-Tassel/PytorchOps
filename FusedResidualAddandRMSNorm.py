import torch

def solve(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    z = x + residual
    rms = torch.sqrt(
        torch.mean(z * z, dim=1, keepdim=True) + eps
    )

    # [N, C] / [N, 1] * [C]
    result = z / rms * weight
    out.copy_(result)