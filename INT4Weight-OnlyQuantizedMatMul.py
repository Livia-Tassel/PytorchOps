import torch

def solve(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_size: int,
):
    packed = w_q.reshape(N, K // 2)

    high = (packed >> 4).to(torch.int16) - 8
    low = (packed & 0x0F).to(torch.int16) - 8

    # [N, K]
    w_int4 = torch.stack(
        (high, low),
        dim=-1
    ).reshape(N, K)

    scale_full = scales.reshape(N, K // group_size).repeat_interleave(group_size, dim=1)

    w = w_int4.to(x.dtype) * scale_full.to(x.dtype)
    result = x.reshape(M, K) @ w.transpose(0, 1)
    y.copy_(result.reshape_as(y))