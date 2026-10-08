import torch

def solve(
    x: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_len: int,
):
    d_model = 512
    n_q_heads = 8
    n_kv_heads = 2
    head_dim = 64
    rope_dim = 32
    eps = 1e-5

    w1 = weights[0:512]

    W_q = weights[512:262656].view(512, 512)
    W_k = weights[262656:328192].view(128, 512)
    W_v = weights[328192:393728].view(128, 512)
    W_o = weights[393728:655872].view(512, 512)

    w2 = weights[655872:656384]

    W_gate = weights[656384:1377280].view(1408, 512)
    W_up = weights[1377280:2098176].view(1408, 512)
    W_down = weights[2098176:2819072].view(512, 1408)

    # RMSNorm
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    x_norm = (x / rms) * w1

    # QKV
    Q = x_norm @ W_q.T
    K = x_norm @ W_k.T
    V = x_norm @ W_v.T

    Q = Q.view(seq_len, n_q_heads, head_dim)
    K = K.view(seq_len, n_kv_heads, head_dim)
    V = V.view(seq_len, n_kv_heads, head_dim)

    # RoPE
    cos_pos = cos[:, None, :]
    sin_pos = sin[:, None, :]

    q1 = Q[..., :rope_dim]
    q2 = Q[..., rope_dim:]

    Q = torch.cat(
        [
            # [L, 8, 64 // 2] * [L, 1, 32]
            q1 * cos_pos - q2 * sin_pos,
            q1 * sin_pos + q2 * cos_pos
        ],
        dim=-1
    )

    k1 = K[..., :rope_dim]
    k2 = K[..., rope_dim:]

    K = torch.cat(
        [
            k1 * cos_pos - k2 * sin_pos,
            k1 * sin_pos + k2 * cos_pos
        ],
        dim=-1
    )

    # GQA
    group_size = n_q_heads // n_kv_heads

    K = K.repeat_interleave(group_size, dim=1)
    V = V.repeat_interleave(group_size, dim=1)

    Q = Q.transpose(0, 1)
    K = K.transpose(0, 1)
    V = V.transpose(0, 1)

    scores = (Q @ K.transpose(1, 2)) / (head_dim ** 0.5)

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device
        ),
        diagonal=1
    )
    scores = scores.masked_fill(
        causal_mask,
        float("-inf")
    )

    attn = torch.softmax(scores, dim=-1)

    heads = attn @ V

    attn_out = heads.transpose(0, 1).contiguous().view(seq_len, d_model)
    attn_out = attn_out @ W_o.T

    # Residual
    x = x + attn_out

    # RMSNorm
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    h_norm = (x / rms) * w2

    # SwiGLU
    gate = h_norm @ W_gate.T
    up = h_norm @ W_up.T

    silu_gate = (
        gate
        * torch.sigmoid(gate)
    )

    hidden = silu_gate * up
    ffn_out = hidden @ W_down.T

    # Residual
    result = x + ffn_out
    output.copy_(result)