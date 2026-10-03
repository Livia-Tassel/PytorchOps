import torch

def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.reshape(n_samples, n_features).to(torch.float32)
    y = y.reshape(n_samples).to(torch.float32)

    # init beta
    b = torch.zeros(
        n_features,
        device=X.device,
        dtype=torch.float32
    )

    lam = 1e-6
    I = torch.eye(
        n_features,
        device=X.device,
        dtype=torch.float32
    )

    for _ in range(20):
        z = X @ b
        p = torch.sigmoid(z)

        # G(beta) = = Xᵀ(y - p) - λbeta
        grad = X.T @ (y - p) - lam * b
        w = p * (1.0 - p)
        # H = XᵀWX + λI
        H = X.T @ (X * w.unsqueeze(1)) + lam * I
        delta = torch.linalg.solve(H, grad)
        b = b + delta
        if torch.max(torch.abs(delta)) < 1e-6:
            break

    beta.view(-1).copy_(b.to(beta.dtype))