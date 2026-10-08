import torch

def solve(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    keep: torch.Tensor,
    N: int,
    iou_threshold: float,
):
    keep.zero_()
    order = torch.argsort(scores, descending=True, stable=True)

    b = boxes[order]
    x1, y1, x2, y2 = b.unbind(dim=1)
    areas = (x2 - x1) * (y2 - y1)

    words = (N + 63) // 64
    padded_n = words * 64

    powers = (
        torch.ones(64, device=boxes.device, dtype=torch.int64)
        << torch.arange(64, device=boxes.device, dtype=torch.int64)
    )

    cols = torch.arange(N, device=boxes.device)

    masks = torch.empty(
        (N, words),
        device=boxes.device,
        dtype=torch.int64
    )

    block = 64
    for start in range(0, N, block):
        end = min(start + block, N)
        rows = torch.arange(start, end, device=boxes.device)

        left = torch.maximum(
            x1[start:end, None], x1[None, :])
        top = torch.maximum(
            y1[start:end, None], y1[None, :])
        right = torch.minimum(
            x2[start:end, None], x2[None, :])
        bottom = torch.minimum(
            y2[start:end, None], y2[None, :])

        inter = (
            (right - left).clamp_min(0)
            * (bottom - top).clamp_min(0))

        union = (
            areas[start:end, None]
            + areas[None, :]
            - inter)

        overlap = (
            (inter > iou_threshold * union)
            & (cols[None, :] > rows[:, None]))

        if padded_n > N:
            overlap = torch.nn.functional.pad(
                overlap, (0, padded_n - N))

        masks[start:end] = (
            overlap.reshape(end - start, words, 64)
            .to(torch.int64) * powers
        ).sum(dim=-1)

    masks_cpu = masks.cpu().numpy()
    suppressed = 0
    chosen = []
    for i in range(N):
        if (suppressed >> i) & 1:
            continue

        chosen.append(i)

        suppressed |= int.from_bytes(
            masks_cpu[i].tobytes(),
            "little",
            signed=False
        )

    if chosen:
        idx = torch.tensor(
            chosen,
            dtype=torch.long,
            device=boxes.device
        )

        keep[order[idx]] = 1