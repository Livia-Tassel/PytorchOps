import torch

def solve(input, output, N, C, H, W, kernel_size, stride, padding):
    x = input.reshape(N, C, H, W)
    # Ho = (H + 2P - K) // S + 1
    if padding > 0:
        padded = torch.full((N, C, H + 2 * padding, W + 2 * padding),
                            float('-inf'),
                            dtype=x.dtype, device=x.device)

        padded[:, :,
               padding:padding + H,
               padding:padding + W
        ] = x
        x = padded

    windows = (
        x.unfold(2, kernel_size, stride)
         .unfold(3, kernel_size, stride)    
    )

    result = windows.amax(dim=(-1, -2))
    output.copy_(result.reshape_as(output))        