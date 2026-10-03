import torch

def solve(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    Y: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    G: int,
    eps: float,
):
    # [N, C, H, W] → [N, G, C/G, H, W]
    X = X.reshape(N, G, C//G, H, W)
    mean = X.mean(dim=(2, 3, 4), keepdim=True)
    var = ((X - mean) ** 2).mean(
        dim=(2, 3, 4),
        keepdim=True
    )

    X_hat = (X - mean) / ((var + eps) ** 0.5)
    X_hat = X_hat.reshape(N, C, H, W)

    Y.copy_(X_hat * gamma.view(1, C, 1, 1)
            + beta.view(1, C, 1, 1)
    )