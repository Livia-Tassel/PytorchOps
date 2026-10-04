import torch

def solve(
    images: torch.Tensor,
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    cls_token: torch.Tensor,
    pos_embed: torch.Tensor,
    output: torch.Tensor,
    B: int,
    C: int,
    H: int,
    W: int,
    P: int,
    D: int,
):
    # [B, C, H, W]
    # → [B, C, H/P, W/P, P, P]
    patches = images.unfold(2, P, P).unfold(3, P, P)

    # [B, H/P, W/P, C, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5)

    N = (H // P) * (W // P)
    # [B, N, C*P*P]
    patches = patches.reshape(B, N, C * P * P)

    weight = patch_weight.reshape(D, C * P * P)

    # [B, N, C*P*P] @ [C*P*P, D]
    # → [B, N, D]
    tokens = patches @ weight.T + patch_bias

    output[:, 0, :] = cls_token
    output[:, 1:, :] = tokens

    output.add_(pos_embed)