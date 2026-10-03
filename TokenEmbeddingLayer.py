import torch

def solve(
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    position_embeddings: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    V: int,
    P: int,
    D: int,
    eps: float,
):
    # [B, T, D]
    token_emb = token_embeddings[token_ids]
    # [T, D]
    pos_emb = position_embeddings[position_ids]
    x = token_emb + pos_emb

    mean = x.mean(
        dim=-1,
        keepdim=True
    )
    var = ((x - mean) ** 2).mean(
        dim=-1,
        keepdim=True
    )
    
    x_hat = (x - mean) / ((var + eps) ** 0.5)
    # [B, T, D] * [D] + [D]
    output.copy_(gamma * x_hat + beta)