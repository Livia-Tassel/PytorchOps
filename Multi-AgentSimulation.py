import torch

def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
    # [x, y, vx, vy]
    a = agents.reshape(N, 4)

    pos = a[:, :2]
    vel = a[:, 2:]

    # [N, N, 2]
    diff = pos[:, None, :] - pos[None, :, :]
    dist = (diff * diff).sum(dim=-1)

    mask = (dist < 25) & (~torch.eye(
        N, device=agents.device, dtype=torch.bool
    ))
    count = mask.sum(dim=1)

    # [N, N] @ [N, 2]
    vel_sum = mask.to(vel.dtype) @ vel
    # [N, 2] / [N, 1]
    avg_vel = vel_sum / count.clamp(min=1).unsqueeze(-1)

    avg_vel = torch.where(
        (count > 0).unsqueeze(-1),
        avg_vel, vel
    )

    new_vel = vel + 0.05 * (avg_vel - vel)
    new_pos = pos + new_vel
    result = torch.cat((new_pos, new_vel), dim=1)
    agents_next.copy_(result.reshape(-1))