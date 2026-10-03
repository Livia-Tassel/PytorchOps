import torch

def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.reshape(n_samples, n_features)
    y = y.reshape(n_samples)
    beta.copy_(torch.linalg.inv(X.T @ X) @ X.T @ y)