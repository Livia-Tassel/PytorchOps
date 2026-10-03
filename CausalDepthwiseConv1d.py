import torch
import torch.nn.functional as F

def solve(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    B: int,
    L: int,
    D: int,
    K: int,
):
    # [B, L, D] → [B, D, L]
    x_cf = x.transpose(1, 2)
    # [D, K] → [D, 1, K]
    w = weight.flip(1).unsqueeze(1)
    x_pad = F.pad(x_cf, (K - 1, 0))

    y = F.conv1d(
        x_pad,
        w,
        bias=bias,
        groups=D
    )
    output.copy_(y.transpose(1, 2))