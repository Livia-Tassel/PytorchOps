import torch

def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    # Xr = (Xq - Zx) * Sx
    A_int = A.view(M, K).to(torch.int32) - zero_point_A
    B_int = B.view(K, N).to(torch.int32) - zero_point_B 
    acc = A_int @ B_int
    C_real = acc.float() * scale_A * scale_B

    C_quant = torch.round(C_real / scale_C) + zero_point_C
    result = torch.clamp(C_quant, min=-128, max=127)
    C.copy_(result)