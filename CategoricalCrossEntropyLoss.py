import torch

def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    # torch.nn.functional.cross_entropy(x)
    probs = torch.softmax(logits, dim=1)
    correct_probs = probs[torch.arange(N), true_labels]
    loss.copy_(-torch.log(correct_probs).mean())