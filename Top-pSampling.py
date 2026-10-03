import torch

def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_indices = probs.sort(descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    remove_mask = cumulative_probs > p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum()

    generator = torch.Generator(device=logits.device)
    generator.manual_seed(int(seed.item()))
    sampled_sorted_index = torch.multinomial(
        filtered_probs,
        num_samples=1,
        generator=generator
    )

    token_id = sorted_indices[sampled_sorted_index]
    sampled_token.copy_(token_id)