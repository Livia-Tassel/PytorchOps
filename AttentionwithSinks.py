import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    num_sinks: int,
    window_size: int,
):
    scores = (Q @ K.T) / (d ** 0.5)

    i = torch.arange(M, device=Q.device)[:, None]
    j = torch.arange(M, device=Q.device)[None, :]
    
    mask = (j <= i) & (
        (j < num_sinks) |
        (j >= i - window_size + 1)
    )

    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output.copy_(weights @ V)