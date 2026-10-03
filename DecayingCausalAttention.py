import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
    d_model: int,
    gamma: float,
):
    idx = torch.arange(seq_len, device=Q.device)
    # dist[n, m] = n - m
    dist = idx[:, None] - idx[None, :]
    # [L, L]
    decay = torch.where(
        dist >= 0,
        gamma ** dist,
        0.0
    )

    # [L, D] @ [D, L] * [L, L]
    scores = (Q @ K.T) / (d_model ** 0.5)
    weights = scores * decay
    output.copy_(weights @ V)