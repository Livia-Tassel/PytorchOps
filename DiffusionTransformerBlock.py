import torch
import torch.nn.functional as F

def solve(
    x: torch.Tensor,
    c: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    seq_len: int,
):
    pos = 0
    def take(shape):
        nonlocal pos
        n = 1
        for size in shape:
            n *= size
        tensor = weights[pos:pos + n].reshape(shape)
        pos += n
        return tensor

    W_ada = take((3072, 512))
    b_ada = take((3072,))
    W_qkv = take((1536, 512))
    b_qkv = take((1536,))
    W_o = take((512, 512))
    b_o = take((512,))
    W_fc1 = take((2048, 512))
    b_fc1 = take((2048,))
    W_fc2 = take((512, 2048))
    b_fc2 = take((512,))

    modulation = F.linear(F.silu(c), W_ada, b_ada)

    shift_msa, scale_msa, gate_msa, \
    shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

    # LayerNorm + Modulation
    h = F.layer_norm(x, (512,), eps=1e-6)
    h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

    # QKV
    qkv = F.linear(h, W_qkv, b_qkv)
    q, k, v = qkv.chunk(3, dim=-1)

    # Attention Heads
    q = q.reshape(batch_size, seq_len, 8, 64).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, 8, 64).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, 8, 64).transpose(1, 2)

    # Multi-Head Self-Attention
    attn = F.scaled_dot_product_attention(
        q, k, v, is_causal=False
    )

    attn = attn.transpose(1, 2).contiguous()
    attn = attn.reshape(batch_size, seq_len, 512)
    attn = F.linear(attn, W_o, b_o)

    # Gate + Residual
    x_prime = x + gate_msa[:, None, :] * attn

    # LayerNorm + Modulation
    h = F.layer_norm(x_prime, (512,), eps=1e-6)
    h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]

    # MLP
    mlp = F.linear(h, W_fc1, b_fc1)
    mlp = F.gelu(mlp, approximate="tanh")
    mlp = F.linear(mlp, W_fc2, b_fc2)

    # Gate + Residual
    result = x_prime + gate_mlp[:, None, :] * mlp
    output.copy_(result)