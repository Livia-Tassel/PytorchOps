import torch

def solve(
    Q: torch.Tensor,
    K_int8: torch.Tensor,
    V_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    # [H, L, D] * [H, L, 1]
    K = K_int8 * k_scale.unsqueeze(-1)
    V = V_int8 * v_scale.unsqueeze(-1)

    # [H, 1, D] @ [H, D, L] → [H, 1, L] → [H, L]
    scores = torch.bmm(
        Q.unsqueeze(1),
        K.transpose(1, 2)
    ).squeeze(1)
    scores = scores / (head_dim ** 0.5)

    weights = torch.softmax(scores, dim=-1)
    # [H, 1, L] @ [H, L, D]
    result = torch.bmm(
        weights.unsqueeze(1),
        V
    ).squeeze(1)
    output.copy_(result)