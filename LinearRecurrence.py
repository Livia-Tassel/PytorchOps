import torch

def solve(a: torch.Tensor, x: torch.Tensor, h: torch.Tensor, B: int, L: int):
    a_scan = torch.cat(
        # [1, a[1], a[2], ..., a[L-1]]
        (torch.ones_like(a[:, :1]), a[:, 1:]),
        dim=1
    )

    # p != 0
    # p[0] = 1
    # p[1] = a[1]
    # p[2] = a[1] * a[2]
    # ...
    p = torch.cumprod(a_scan, dim=1)

    # h = p * cumsum(x / p)
    result = p * torch.cumsum(x / p, dim=1)
    h.copy_(result)