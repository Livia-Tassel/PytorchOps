import torch

def solve(
    beam_scores: torch.Tensor,
    token_logprobs: torch.Tensor,
    new_beam_scores: torch.Tensor,
    parent_beam_indices: torch.Tensor,
    next_tokens: torch.Tensor,
    B: int,
    K: int,
    V: int,
):
    # [B, K] → [B, K, 1]
    beam_scores = beam_scores.unsqueeze(-1)
    # [B, K, 1] + [B, K, V]
    candidates = beam_scores + token_logprobs
    candidates = candidates.reshape(B, K * V)

    top_scores, top_indices = torch.topk(
        candidates,
        K,
        dim=1
    )

    new_beam_scores.copy_(top_scores)
    parent_indices = top_indices // V
    token_indices = top_indices % V
    parent_beam_indices.copy_(parent_indices)
    next_tokens.copy_(token_indices)