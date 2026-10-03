import torch

def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    mse.copy_(((predictions - targets) ** 2).mean())