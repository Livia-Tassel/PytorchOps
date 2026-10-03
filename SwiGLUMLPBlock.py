import torch

def solve(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d_model: int,
    d_ffn: int,
):
    # [M, d_model] @ [d_model, d_ffn]
    gate = x @ W_gate
    up = x @ W_up
    # SiLU
    gate = gate * torch.sigmoid(gate)
    hidden = gate * up

    # [M, d_ffn] @ [d_ffn, d_model]
    result = hidden @ W_down
    output.copy_(result)