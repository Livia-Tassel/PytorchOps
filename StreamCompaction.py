import torch

def solve(A: torch.Tensor, N: int, out: torch.Tensor):
    mask = A > 0.0
    compacted = A[mask]
    out.fill_(0.0)
    out[:compacted.numel()].copy_(compacted)