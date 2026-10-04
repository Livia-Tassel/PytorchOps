import torch

def solve(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    Q_out: torch.Tensor,
    B: int,
    d_model: int,
    H_q: int,
    H_kv: int,
    D: int,
    S_max: int,
):
    # [B, d] @ [d, (H_q + 2*H_kv)*D]
    qkv = x @ W_qkv

    q_end = H_q * D
    k_end = q_end + H_kv * D

    q = qkv[:, :q_end]
    k = qkv[:, q_end:k_end]
    v = qkv[:, k_end:]

    k = k.reshape(B, H_kv, D)
    v = v.reshape(B, H_kv, D)
    q = q.reshape(B, H_q, D)

    # [B, D]
    rope = cos_sin_cache[positions]
    # [B, D/2] -> [B, 1, D/2]
    cos = rope[:, :D // 2].unsqueeze(1)
    sin = rope[:, D // 2:].unsqueeze(1)

    q_first = q[..., :D // 2]
    q_second = q[..., D // 2:]

    q_rot = torch.cat((
            q_first * cos - q_second * sin,
            q_second * cos + q_first * sin,
        ),
        dim=-1,
    )

    k_first = k[..., :D // 2]
    k_second = k[..., D // 2:]

    k_rot = torch.cat((
            k_first * cos - k_second * sin,
            k_second * cos + k_first * sin,
        ),
        dim=-1,
    )

    Q_out.copy_(q_rot)
    batch_idx = torch.arange(B, device=x.device)

    # [B, H_kv, p, D] → [B, H_kv, S, D]
    K_cache[batch_idx, :, positions, :] = k_rot
    V_cache[batch_idx, :, positions, :] = v