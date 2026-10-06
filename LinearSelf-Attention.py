import torch

def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    phi_Q = torch.where(Q > 0, Q + 1, torch.exp(Q))
    phi_K = torch.where(K > 0, K + 1, torch.exp(K))

    # [d, M] @ [M, d] 
    KV = phi_K.T @ V
    # [M, d] @ [d, d]
    numerator = phi_Q @ KV

    # [d]
    K_sum = phi_K.sum(dim=0)
    # [M, d] @ [d] → [M]
    denominator = phi_Q @ K_sum

    # [M, d] / [M, 1]
    result = numerator / denominator[:, None]
    output.copy_(result)