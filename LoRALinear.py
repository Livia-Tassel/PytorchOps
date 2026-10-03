import torch

def solve(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
):
    base = x @ W.T
    lora = (x @ A.T) @ B.T
    result = base + lora_scale * lora
    output.copy_(result)