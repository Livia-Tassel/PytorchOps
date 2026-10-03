import torch

def solve(signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
    x = signal.reshape(M, N, 2)
    real = x[:, :, 0]
    imag = x[:, :, 1]

    # [0, 1, ..., N-1]
    n = torch.arange(
        N,
        device=signal.device,
        dtype=signal.dtype
    )
    k = torch.arange(
        N,
        device=signal.device,
        dtype=signal.dtype
    )

    # angle[k, n] = -2πkn/N
    angle = -2.0 * torch.pi * k[:, None] * n[None, :] / N
    cos_n = torch.cos(angle)
    sin_n = torch.sin(angle)

    row_real = real @ cos_n.T - imag @ sin_n.T
    row_imag = real @ sin_n.T + imag @ cos_n.T

    m = torch.arange(
        M,
        device=signal.device,
        dtype=signal.dtype
    )
    u = torch.arange(
        M,
        device=signal.device,
        dtype=signal.dtype
    )

    # angle[u, m] = -2πum/M
    angle = -2.0 * torch.pi * u[:, None] * m[None, :] / M
    cos_m = torch.cos(angle)
    sin_m = torch.sin(angle)

    out_real = cos_m @ row_real - sin_m @ row_imag
    out_imag = sin_m @ row_real + cos_m @ row_imag

    out = torch.stack(
        (out_real, out_imag),
        dim=-1
    ).reshape(-1)
    spectrum.copy_(out)