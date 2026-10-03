import torch

def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    T: int,
    d: int,
    S: int,
):
    scale = d ** -0.5
    for s in range(S):
        start = int(cu_seqlens[s].item())
        end = int(cu_seqlens[s+1].item())
        L = end - start

        # ✓ . . | . . | .
        # ✓ ✓ . | . . | .
        # ✓ ✓ ✓ | . . | .
        # ------+-----+--
        # . . . | ✓ . | .
        # . . . | ✓ ✓ | .
        # ------+-----+--
        # . . . | . . | ✓
        q = Q[start:end]
        k = K[start:end]
        v = V[start:end]

        # [L, d] @ [d, L]
        scores = (q @ k.transpose(0, 1)) * scale

        causal_mask = torch.triu(
            torch.ones(
                (L, L),
                device=Q.device,
                dtype=torch.bool
            ),
            diagonal=1
        )
        scores.masked_fill_(
            causal_mask,
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)
        result = weights @ v
        output[start:end].copy_(result)