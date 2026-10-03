import torch

def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    logits = logits.view(M, E)
    values, indices = torch.topk(logits, k, dim=1)
    weights = torch.softmax(values, dim=1)

    topk_weights.copy_(weights)
    topk_indices.copy_(indices)