import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    # Q: [M, d]
    # K: [N, d]
    # V: [N, d]
    scores = Q @ K.T / (d ** 0.5)

    i = torch.arange(M, device=Q.device).reshape(M, 1)
    j = torch.arange(N, device=Q.device).reshape(1, N)
    # [M, N]
    delta = i - j

    scores = scores + alpha * delta
    weights = torch.softmax(scores, dim=1)
    result = weights @ V
    output.copy_(result)