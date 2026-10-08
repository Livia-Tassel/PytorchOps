import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int,
    num_selected: int
):
    num_blocks = (seq_len + block_size - 1) // block_size

    if num_selected >= num_blocks:
        # [H, T, D] @ [H, D, 1]
        scores = torch.bmm(K, Q.unsqueeze(-1)).squeeze(-1)
        scores = scores / (head_dim ** 0.5)

        attn = torch.softmax(scores, dim=-1)
        # [H, 1, T] @ [H, T, D]
        result = torch.bmm(
            attn.unsqueeze(1), V).squeeze(1)

        output.copy_(result)
        return

    padded_len = num_blocks * block_size
    pad = padded_len - seq_len

    K_padded = torch.nn.functional.pad(
        K, (0, 0, 0, pad)
    )

    # [H, M, B, D]
    K_blocks = K_padded.reshape(
        num_heads, num_blocks, block_size, head_dim
    )

    lengths = torch.full(
        (num_blocks,), block_size,
        device=K.device,
        dtype=torch.long
    )
    lengths[-1] = seq_len - (num_blocks - 1) * block_size

    # [H, M, B, D] → [H, M, D] / [1, M, 1]
    mean_K = K_blocks.sum(dim=2) / lengths[None, :, None]

    # [H, M, D] @ [H, D, 1]
    block_scores = torch.bmm(mean_K, Q.unsqueeze(-1)).squeeze(-1)
    # [H, M] → [H, S]
    selected = torch.argsort(
        block_scores,
        dim=-1,
        descending=True,
        stable=True
    )[:, :num_selected]

    offsets = torch.arange(block_size, device=K.device)
    # block offset → token offset
    positions = (
        selected[:, :, None] * block_size
        + offsets[None, None, :]
    )

    # [H, S, B] → [H, SB]
    positions = positions.reshape(num_heads, -1)
    valid = positions < seq_len

    safe_positions = positions.clamp(max=seq_len-1)
    head_indices = torch.arange(
        num_heads,
        device=K.device
    )[:, None]
    K_selected = K[head_indices, safe_positions]
    V_selected = V[head_indices, safe_positions]

    scores = (torch.bmm(
        K_selected, Q.unsqueeze(-1)).squeeze(-1)
        / (head_dim ** 0.5))

    scores = scores.masked_fill(~valid, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    # [H, 1, S] @ [H, S, D]
    result = torch.bmm(attn.unsqueeze(1), V_selected).squeeze(1)

    output.copy_(result)