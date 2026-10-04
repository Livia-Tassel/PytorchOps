import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    dQ: torch.Tensor,
    dK: torch.Tensor,
    dV: torch.Tensor,
    M: int,
    N: int,
    d: int,
):
    scores = (Q @ K.T) / (d ** 0.5)
    P = torch.softmax(scores, dim=1)

    # [M, d] @ [d, N]
    dP = dO @ V.T
    # [N, M] @ [M, d]
    dV_result = P.T @ dO

    row_sum = (P * dP).sum(dim=1, keepdim=True)
    dS = P * (dP - row_sum)
    dS = dS / (d ** 0.5)
    dQ_result = dS @ K
    dK_result = dS.T @ Q

    dQ.copy_(dQ_result)
    dK.copy_(dK_result)
    dV.copy_(dV_result)