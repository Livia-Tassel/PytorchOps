import torch

def solve(dist: torch.Tensor, output: torch.Tensor, N: int):
    D = dist.view(N, N).clone()

    for k in range(N):
        D = torch.minimum(
            D,
            # i → k, k → j
            D[:, k:k+1] + D[k:k+1, :]
        )

    output.copy_(D.reshape(-1))