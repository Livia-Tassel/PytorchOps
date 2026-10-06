import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    d_k = d_model // h
    # [h, N, d_k]
    q = Q.reshape(N, h, d_k).transpose(0, 1)
    k = K.reshape(N, h, d_k).transpose(0, 1)
    v = V.reshape(N, h, d_k).transpose(0, 1)

    # [h, N, d_k] @ [h, d_k, N]
    scores = torch.matmul(
        q,
        k.transpose(1, 2)
    ) / (d_k ** 0.5)
    
    attn = torch.softmax(scores, dim=-1)
    heads = attn @ v

    result = heads.transpose(0, 1).reshape(N, d_model)
    output.copy_(result)