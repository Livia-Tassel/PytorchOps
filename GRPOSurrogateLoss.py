import torch

def solve(
    rewards: torch.Tensor,
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    log_ref: torch.Tensor,
    output: torch.Tensor,
    clip_eps: float,
    beta: float,
    B: int,
    G: int,
    S: int,
):
    # [B, G] → [B, 1]
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(
        dim=1,
        keepdim=True,
        correction=0
    )

    advantages = (rewards - mean) / (std + 1e-8)
    # [B, G] → [B, G, 1]
    advantages = advantages.unsqueeze(-1)

    ratio = torch.exp(log_pi - log_pi_old)
    ratio_clipped = torch.clamp(
        ratio,
        min=1-clip_eps,
        max=1+clip_eps

    )

    # [B, G, S]
    surrogate = torch.minimum(
        ratio * advantages,
        ratio_clipped * advantages
    )

    d = log_ref - log_pi
    kl = torch.exp(d) - d - 1
    loss = -(surrogate - beta * kl).mean()
    output[0] = loss