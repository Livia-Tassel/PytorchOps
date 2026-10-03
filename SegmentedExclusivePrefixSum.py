import torch

def solve(values: torch.Tensor, flags: torch.Tensor, output: torch.Tensor, N: int):
    prefix = torch.cumsum(values, dim=0)
    exclusive = prefix - values

    segment_ids = torch.cumsum(flags, dim=0) - 1
    starts = torch.nonzero(flags, as_tuple=False).squeeze(1)

    offsets = torch.zeros(
        starts.numel(),
        device=values.device,
        dtype=values.dtype
    )
    mask = starts > 0
    offsets[mask] = exclusive[starts[mask]]
    result = exclusive - offsets[segment_ids]
    output.copy_(result)