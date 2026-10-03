import torch

def solve(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    result.copy_((b - a) * torch.mean(y_samples))