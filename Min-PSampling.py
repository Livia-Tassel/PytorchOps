import torch

def solve(logits: torch.Tensor, probs: torch.Tensor, min_p: float, B: int, V: int):
    p = torch.softmax(logits, dim=1)

    # [B, V] → [B, 1]
    max_probs = p.max(dim=1, keepdim=True).values
    threshold = max_probs * min_p
    filtered = torch.where(
        p >= threshold,
        p,
        0.0
    )

    filtered = filtered / filtered.sum(dim=1, keepdim=True)
    probs.copy_(filtered)