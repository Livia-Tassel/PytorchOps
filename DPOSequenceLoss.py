import torch

def solve(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_ref_logps: torch.Tensor,
    rejected_ref_logps: torch.Tensor,
    output: torch.Tensor,
    beta: float,
    B: int,
):
    policy_margin = chosen_logps - rejected_logps
    ref_margin = chosen_ref_logps - rejected_ref_logps

    logits = beta * (policy_margin - ref_margin)
    loss = torch.nn.functional.softplus(-logits)
    result = loss.mean()
    output[0] = result