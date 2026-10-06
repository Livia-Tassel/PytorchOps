import torch

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    if N <= 0:
        return

    cur = input.to(torch.int64)
    buf = torch.empty_like(cur)

    for shift in range(32):
        bits = (cur >> shift) & 1
        is_zero = 1 - bits

        zero_prefix = torch.cumsum(
            is_zero, dim=0
        )
        one_prefix = torch.cumsum(bits, dim=0)
        num_zeros = zero_prefix[-1]

        # 0-based
        zero_rank = zero_prefix - 1
        one_rank = one_prefix - 1

        positions = torch.where(
            bits == 0,
            zero_rank,
            num_zeros + one_rank
        )

        buf[positions] = cur
        cur, buf = buf, cur

    output.copy_(cur.to(output.dtype))