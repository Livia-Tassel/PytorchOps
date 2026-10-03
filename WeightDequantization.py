import torch

def solve(X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int):
    X = X.view(M, N)
    S = S.view(
        (M + TILE_SIZE - 1) // TILE_SIZE,
        (N + TILE_SIZE - 1) // TILE_SIZE
    )

    S = (
        S.repeat_interleave(TILE_SIZE, dim=0)
         .repeat_interleave(TILE_SIZE, dim=1)
    )
    S = S[:M, :N]
    Y.copy_((X * S).reshape_as(Y))