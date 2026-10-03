import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
):
    group_size = num_q_heads // num_kv_heads

    Q = Q.reshape(
        num_q_heads,
        seq_len,
        head_dim
    )
    K = K.reshape(num_kv_heads, seq_len, head_dim)
    V = V.reshape(num_kv_heads, seq_len, head_dim)

    K = K.repeat_interleave(
        group_size, dim=0
    )
    V = V.repeat_interleave(
        group_size, dim=0
    )

    scores = torch.bmm(Q, K.transpose(1, 2))
    scores = scores / (head_dim ** 0.5)

    weights = torch.softmax(scores, dim=-1)
    result = torch.bmm(weights, V)
    output.copy_(result.reshape_as(output))