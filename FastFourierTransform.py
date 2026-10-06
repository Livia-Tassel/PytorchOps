import torch
import math

def _fft_radix(x: torch.Tensor, inverse: bool = False):
    N = x.numel()

    if N <= 1:
        return x

    device = x.device
    bits = N.bit_length() - 1
    idx = torch.arange(
        N,
        device=device,
        dtype=torch.int64
    )

    rev = torch.zeros_like(idx)
    tmp = idx.clone()

    for _ in range(bits):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1

    x = x[rev].clone()

    size = 2

    while size <= N:
        half = size // 2
        groups = N // size

        j = torch.arange(
            half,
            device=device,
            dtype=torch.float32
        )

        angle = 2.0 * math.pi * j / size

        if not inverse:
            angle = -angle

        # W = cos(theta) + j sin(theta)
        wr = torch.cos(angle)
        wi = torch.sin(angle)

        w = torch.complex(wr, wi)

        y = x.view(groups, size)

        a = y[:, :half].clone()
        b = y[:, half:].clone()

        t = b * w

        y[:, :half] = a + t
        y[:, half:] = a - t

        size *= 2

    if inverse:
        x /= N

    return x

def solve(signal: torch.Tensor, spectrum: torch.Tensor, N: int):
    if N <= 0:
        return

    inp = signal.view(N, 2)

    x = torch.complex(
        inp[:, 0],
        inp[:, 1]
    )

    if (N & (N - 1)) == 0:
        result = _fft_radix(x)

        out = spectrum.view(N, 2)

        out[:, 0].copy_(result.real)
        out[:, 1].copy_(result.imag)

        return

    device = signal.device

    # M >= 2N - 1
    M = 1 << (2 * N - 1).bit_length()

    n_int = torch.arange(
        N,
        device=device,
        dtype=torch.int64
    )

    phase_index = (n_int * n_int) % (2 * N)
    phase = (
        math.pi
        * phase_index.to(torch.float32)
        / N
    )

    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)

    chirp_pos = torch.complex(
        cos_phase,
        sin_phase
    )

    chirp_neg = torch.conj(chirp_pos)

    # a[n] = x[n] * exp(-j*pi*n²/N)
    a = torch.zeros(
        M,
        device=device,
        dtype=torch.complex64
    )

    a[:N] = x * chirp_neg

    # b[n] = exp(+j*pi*n²/N)
    # b[-n] = b[n]
    b = torch.zeros(
        M,
        device=device,
        dtype=torch.complex64
    )

    b[:N] = chirp_pos

    if N > 1:
        b[M - N + 1:] = torch.flip(
            chirp_pos[1:],
            dims=[0]
        )

    A = _fft_radix(a)
    B = _fft_radix(b)
    C = A * B

    c = _fft_radix(
        C,
        inverse=True
    )

    result = c[:N] * chirp_neg
    out = spectrum.view(N, 2)

    out[:, 0].copy_(result.real)
    out[:, 1].copy_(result.imag)