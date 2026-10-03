import torch
import math

def solve(
    prompts: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int
):
    # Model Parameters
    w0, w1 = weights[0], weights[1]
    q0, q1 = weights[2], weights[3]
    v0 = weights[4]
    a, c = weights[5], weights[6]
    carry = weights[7]
    n0, n1 = weights[8], weights[9]

    eps = 1e-6
    omega = 2.0 * math.pi / 19.0

    # Attention Scale
    S2 = math.log(10.0) / (
        math.sqrt(2.0)
        * (
            math.cos(0.3 * omega)
            - math.cos(0.7 * omega)
        )
    )
    attn_scale = S2 / math.sqrt(2.0)

    device = prompts.device
    dtype = weights.dtype

    # Embedding Table
    # e(d) = [w0 - w1*d^2, -d]
    digits = torch.arange(
        10,
        device=device,
        dtype=dtype
    )

    E = torch.stack((
            w0 - w1 * digits * digits,
            -digits
        ),
        dim=-1
    )

    # [B, L]
    tokens = prompts
    # Autoregressive Decoding
    for step in range(11):
        seq_len = tokens.shape[1]
        # [B, L, 2]
        x = E[tokens.long()]
        residual = x

        # Unit RMSNorm
        # [B, L, 1]
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + eps
        )

        # x / sqrt(mean(x^2) + eps)
        h = x / rms

        h0 = h[:, :, 0]
        h1 = h[:, :, 1]

        # Q = [h0*q0, h0*q1]
        # K = [h0, 0]
        # V = [h1*v0, 0]
        Q = torch.stack((
                h0 * q0,
                h0 * q1
            ),
            dim=-1
        )

        K = torch.stack((
                h0,
                torch.zeros_like(h0)
            ),
            dim=-1
        )

        V = torch.stack((
                h1 * v0,
                torch.zeros_like(h1)
            ),
            dim=-1
        )

        # QK-Norm
        q_rms = torch.sqrt(
            torch.mean(Q * Q, dim=-1, keepdim=True)
            + eps
        )

        k_rms = torch.sqrt(
            torch.mean(K * K, dim=-1, keepdim=True)
            + eps
        )

        Q = Q / q_rms
        K = K / k_rms

        # RoPE
        # [0, 1, 2, ..., L-1]
        pos = torch.arange(
            seq_len,
            device=device,
            dtype=dtype
        )

        angle = pos * omega

        # [L] → [1, L]
        cos = torch.cos(angle)[None, :]
        sin = torch.sin(angle)[None, :]

        Q0 = Q[:, :, 0]
        Q1 = Q[:, :, 1]

        Q = torch.stack((
                Q0 * cos - Q1 * sin,
                Q0 * sin + Q1 * cos
            ),
            dim=-1
        )

        K0 = K[..., 0]
        K1 = K[..., 1]

        K = torch.stack((
                K0 * cos - K1 * sin,
                K0 * sin + K1 * cos
            ),
            dim=-1
        )

        # Scaled Dot-Product Attention
        # Q: [B, L, 2]
        # K: [B, L, 2]
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )
        scores = scores * attn_scale

        # Causal Mask
        # [B, L, L]
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        # Softmax
        probs = torch.softmax(
            scores,
            dim=-1
        )

        # Attention @ V
        # [B,L,L] @ [B,L,2] → [B,L,2]
        attn = torch.matmul(
            probs,
            V
        )

        # [attn0, 0]
        #      ↓
        # [0, attn0]
        attn_out = torch.stack((
                torch.zeros_like(attn[..., 0]),
                attn[..., 0]
            ),
            dim=-1
        )

        # Residual Connection
        x = residual + attn_out

        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + eps
        )
        h = x / rms

        h0 = h[..., 0]
        h1 = h[..., 1]

        # MLP
        g0 = h0 * a + h1 * c
        g1 = h0 * (a - c / 1000.0) + h1 * c

        # SiLU(x) = x * sigmoid(x)
        silu_g0 = g0 * torch.sigmoid(g0)
        silu_g1 = g1 * torch.sigmoid(g1)

        base = h0
        mix0 = silu_g0 * base
        mix1 = silu_g1 * base

        # MLP(h) = [0, carry * (mix1 - mix0)]
        mlp_out = torch.stack((
                torch.zeros_like(mix0),
                carry * (mix1 - mix0)
            ),
            dim=-1
        )

        x = x + mlp_out

        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + eps
        )
        x = x / rms

        norm_weight = torch.stack((n0, n1))
        x = x * norm_weight

        # Output Logits
        # [B,L,2] @ [2,10] → [B,L,10]
        logits = torch.matmul(
            x,
            E.transpose(0, 1)
        )

        # [B, 10]
        last_logits = logits[:, -1, :]
        output[:, step, :].copy_(last_logits)

        next_token = torch.argmax(
            last_logits,
            dim=-1
        )
        tokens = torch.cat((
                tokens,
                # [B] → [B, 1]
                next_token[:, None]
            ),
            dim=1
        )