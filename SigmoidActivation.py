import torch

def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    Y.copy_(torch.sigmoid(X))