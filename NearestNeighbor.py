import torch

def solve(points: torch.Tensor, indices: torch.Tensor, N: int):
    points = points.view(N, 3)
    chunk_size = 1024

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = points[start:end]

        # [chunk_size, N, 3]
        diff = chunk[:, None, :] - points[None, :, :]
        # [chunk_size, N]
        dist = (diff * diff).sum(dim=2)

        # [0, 1, 2, ..., (end-start)-1]
        rows = torch.arange(end - start, device=points.device)
        # [start, ..., end]
        cols = torch.arange(start, end, device=points.device)

        dist[rows, cols] = float('inf')
        nearest = torch.argmin(dist, dim=1)
        indices[start:end].copy_(nearest)