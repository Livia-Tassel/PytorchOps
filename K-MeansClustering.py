import torch

def solve(
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    labels: torch.Tensor,
    initial_centroid_x: torch.Tensor,
    initial_centroid_y: torch.Tensor,
    final_centroid_x: torch.Tensor,
    final_centroid_y: torch.Tensor,
    sample_size: int,
    k: int,
    max_iterations: int,
):
    cx = initial_centroid_x.clone()
    cy = initial_centroid_y.clone()

    for _ in range(max_iterations):
        # [N, k]
        dx = data_x[:, None] - cx[None, :]
        dy = data_y[:, None] - cy[None, :]
        dist = dx * dx + dy * dy

        current_labels = dist.argmin(dim=1)

        sum_x = torch.zeros(
            k, device=data_x.device, dtype=data_x.dtype
        )
        sum_y = torch.zeros(
            k, device=data_y.device, dtype=data_y.dtype
        )
        counts = torch.zeros(
            k, device=data_x.device, dtype=data_x.dtype
        )

        # (dim, idx, src)
        sum_x.scatter_add_(0, current_labels, data_x)
        sum_y.scatter_add_(0, current_labels, data_y)
        counts.scatter_add_(
            0,
            current_labels,
            torch.ones_like(data_x)
        )

        nonempty = counts > 0

        cx = torch.where(
            nonempty,
            sum_x / counts.clamp(min=1),
            cx
        )

        cy = torch.where(
            nonempty,
            sum_y / counts.clamp(min=1),
            cy
        )

    labels.copy_(current_labels.to(labels.dtype))
    final_centroid_x.copy_(cx)
    final_centroid_y.copy_(cy)