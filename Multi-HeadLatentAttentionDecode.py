import torch

def solve(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    kv_lora_rank: int,
    head_dim: int,
    rope_dim: int,
):
    # [H, D+R]
    # → [H, D] + [H, R]
    q_nope = q[:, :head_dim]
    q_pe = q[:, head_dim:head_dim + rope_dim]

    # [T, C+R]
    # → [T, C] + [T, R]
    c = kv_cache[:seq_len, :kv_lora_rank]
    k_pe = kv_cache[:seq_len, kv_lora_rank:kv_lora_rank + rope_dim]

    # [H, 1, D] @ [H, D, C]
    q_latent = torch.bmm(q_nope.unsqueeze(1), W_UK).squeeze(1)
    # [H, C] @ [C, T]
    scores_nope = q_latent @ c.T
    scores_rope = q_pe @ k_pe.T

    scores = (scores_nope + scores_rope) / ((head_dim + rope_dim) ** 0.5)

    attn = torch.softmax(scores, dim=-1)
    latent_out = attn @ c
    # [H, 1, C] @ [H, C, D]
    result = torch.bmm(latent_out.unsqueeze(1), W_UV).squeeze(1)

    output.copy_(result)