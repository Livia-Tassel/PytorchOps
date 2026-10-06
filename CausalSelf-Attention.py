import torch

def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    scores = Q @ K.T / (d ** 0.5)

    mask = torch.triu(
        torch.ones(
            M, M,
            device=Q.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=1)

    result = weights @ V
    output.copy_(result)