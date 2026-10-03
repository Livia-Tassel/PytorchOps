import torch

def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int):
    pos_A = (
        torch.arange(M, device=A.device) +
        torch.searchsorted(B, A, right=False)
    )
    pos_B = (
        torch.arange(N, device=B.device) +
        torch.searchsorted(A, B, right=True)
    )

    # A = [1, 3, 3]
    # B = [2, 3, 4]
    # C = [1, 2, 3, 3, 3, 4]
    C[pos_A] = A
    C[pos_B] = B