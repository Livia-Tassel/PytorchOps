import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    H: int,
    D: int,
):
    # Q: [M, H, D]
    # K: [N, H, D]
    # V: [N, H, D]
    Q = Q.transpose(0, 1)
    K = K.transpose(0, 1)
    V = V.transpose(0, 1)

    # [H, M, D] @ [H, D, N]
    scores = (Q @ K.transpose(1, 2)) / (D ** 0.5)
    weights = torch.softmax(
        scores, dim=-1
    )
    result = (weights @ V).transpose(0, 1)
    output.copy_(result)