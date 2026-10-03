import torch

def solve(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    pixels = input.view(-1, 3)
    f = torch.tensor([0.299, 0.587, 0.114], dtype=input.dtype, device=input.device)
    output.copy_(torch.matmul(pixels, f).view_as(output))