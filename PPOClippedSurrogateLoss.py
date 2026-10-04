import torch

def solve(
    advantages: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    B: int,
    S: int,
):
    # [B, S]
    ratio = torch.exp(log_pi - log_pi_old)
    ratio_clipped = torch.clamp(
        ratio,
        min=1-clip_eps,
        max=1+clip_eps
    )

    # minmum([B, S], [B, S]) → [B, S]
    surrogate = torch.minimum(
        ratio * advantages,
        ratio_clipped * advantages
    )
    result = -surrogate.mean()
    output[0] = result