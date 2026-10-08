import torch

def solve(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, seq_len: int):
    d_model = 768
    n_heads = 12
    head_dim = 64
    ffn_dim = 3072
    eps = 1e-5

    g1 = weights[0:768]
    b1 = weights[768:1536]

    W_qkv = weights[1536:1771008].view(768, 2304)
    b_qkv = weights[1771008:1773312]

    W_attn = weights[1773312:2363136].view(768, 768)
    b_attn = weights[2363136:2363904]

    g2 = weights[2363904:2364672]
    b2 = weights[2364672:2365440]

    W_fc = weights[2365440:4724736].view(768, 3072)
    b_fc = weights[4724736:4727808]

    W_proj = weights[4727808:7087104].view(3072, 768)
    b_proj = weights[7087104:7087872]

    # LayerNorm
    # [L, d]
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

    # QKV
    x_norm = ((x - mean) / torch.sqrt(var + eps)) * g1 + b1
    qkv = x_norm @ W_qkv + b_qkv
    Q, K, V = qkv.chunk(3, dim=-1)

    Q = Q.view(seq_len, n_heads, head_dim).transpose(0, 1)
    K = K.view(seq_len, n_heads, head_dim).transpose(0, 1)
    V = V.view(seq_len, n_heads, head_dim).transpose(0, 1)

    # MHA
    scores = (Q @ K.transpose(1, 2)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    heads = attn @ V

    attn_out = heads.transpose(0, 1).contiguous().view(seq_len, d_model)
    attn_out = attn_out @ W_attn + b_attn

    # Residual
    x = x + attn_out

    # LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

    # FFN
    h_norm = ((x - mean) / torch.sqrt(var + eps)) * g2 + b2
    # up 768 → 3072
    h = h_norm @ W_fc + b_fc
    h = 0.5 * h *(
        1.0
        + torch.tanh(
            (2.0 / torch.pi) ** 0.5
            * (h + 0.044715 * h ** 3)
        )
    )
    # down 3072 → 768
    ffn_out = h @ W_proj + b_proj

    result = x + ffn_out
    output.copy_(result)