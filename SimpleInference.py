import torch
import torch.nn as nn

def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    # X: [N, K]
    # model : nn.Linear(K, M)
    # Y: [N, M] 
    output.copy_(model(input))