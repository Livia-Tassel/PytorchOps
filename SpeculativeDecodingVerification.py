import torch

def solve(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    output_tokens: torch.Tensor,
    B: int,
    T: int,
    V: int,
):
    output_tokens.zero_()

    for b in range(B):
        all_accepted = True
        for i in range(T):
            # batch b's i-th suggestion token
            token = draft_tokens[b, i].long()
            # [B, T, V]
            p = draft_probs[b, i, token]
            q = target_probs[b, i, token]

            if p == 0:
                # acceptance probability
                alpha = 1.0 if q > 0 else 0.0
            else:
                alpha = min(1.0, (q / p).item())

            # [α, 1-α]
            if uniform_samples[b, i].item() < alpha:
                output_tokens[b, i] = token
            else:
                all_accepted = False
                # sample from the remaining tokens
                adjusted = torch.clamp(
                    target_probs[b, i] - draft_probs[b, i],
                    min=0.0
                )

                total = adjusted.sum()
                if total.item() > 0:
                    probs = adjusted / total
                else:
                    probs = target_probs[b, i]

                cdf = torch.cumsum(probs, dim=0)
                u = uniform_samples[b, T]
                # [  |  | u |  ]
                replacement = torch.searchsorted(
                    cdf, u, right=False
                )
                replacement = torch.clamp(
                    replacement, max=V-1
                )

                output_tokens[b, i] = replacement.to(
                    output_tokens.dtype
                )

                break

        if all_accepted:
            probs = target_probs[b, T-1]
            cdf = torch.cumsum(probs, dim=0)
            u = uniform_samples[b, T]

            bonus = torch.searchsorted(
                cdf, u, right=False
            )
            bonus = torch.clamp(
                bonus, max=V-1
            )

            output_tokens[b, T] = bonus.to(
                output_tokens.dtype
            )