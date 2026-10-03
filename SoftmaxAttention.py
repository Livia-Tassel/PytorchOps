import torch

def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    # softmax(Q @ K / sqrt(d)) @ V
    scores = Q @ K.T / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    output.copy_(attn @ V)