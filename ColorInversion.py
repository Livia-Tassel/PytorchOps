import torch

def solve(image: torch.Tensor, width: int, height: int):
    # image: [width, height, 4] 1D
    # 4 → [R, G, B, A]
    pixels = image.view(-1, 4)
    pixels[:, :3].copy_(255 - pixels[:, :3])