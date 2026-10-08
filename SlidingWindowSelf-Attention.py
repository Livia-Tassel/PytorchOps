import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    scores = (Q @ K.T) / (d ** 0.5)

    i = torch.arange(M, device=Q.device)[:, None]
    j = torch.arange(M, device=Q.device)[None, :]

    mask = (j >= i - window_size) & (j <= i + window_size)
    scores = torch.where(
        mask > 0,
        scores,
        float("-inf")
    )

    weights = torch.softmax(scores, dim=-1)
    result = weights @ V
    output.copy_(result)