import torch

def solve(
    rewards: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    lam: float,
    B: int,
    S: int,
):
    # delta_t = r_t + gamma * V_{t+1} - V_t
    delta = rewards - values
    delta[:, :-1] += gamma * values[:, 1:]

    c = gamma * lam

    gae = torch.zeros(
        B,
        device=rewards.device,
        dtype=torch.float32
    )

    for t in range(S - 1, -1, -1):
        gae = delta[:, t] + c * gae
        advantages[:, t] = gae