import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexAttention(nn.Module):
    """
    Complex-Aware Attention: Respects geometric relationships.
    Score = Re(Q * K_conj) = Qr*Kr + Qi*Ki
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # We need separate projections for Real (Mag*Cos) and Imag (Mag*Sin) parts?
        # Or do we project Mag and Phase directly?
        # Standard approach: Treat Real and Imag as two channels in the linear layer.
        # But here we want to preserve the "meaning" of Mag/Phase.
        # Let's project in Cartesian space (Real/Imag) because linear algebra works better there.

        self.query_real = nn.Linear(d_model, d_model)
        self.query_imag = nn.Linear(d_model, d_model)
        self.key_real = nn.Linear(d_model, d_model)
        self.key_imag = nn.Linear(d_model, d_model)
        self.value_real = nn.Linear(d_model, d_model)
        self.value_imag = nn.Linear(d_model, d_model)

        self.out_real = nn.Linear(d_model, d_model)
        self.out_imag = nn.Linear(d_model, d_model)

    def forward(self, x_real, x_imag, mask=None):
        B, T, C = x_real.shape

        # 1. Projections
        # Q = Qr + iQi
        qr = self.query_real(x_real) - self.query_imag(
            x_imag
        )  # Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        qi = self.query_real(x_imag) + self.query_imag(
            x_real
        )  # Wait, standard linear layers don't do complex mul unless we force it.
        # SIMPLER: Independent projections for components
        qr = self.query_real(x_real)
        qi = self.query_imag(x_imag)  # This effectively mixes them if we train it so.
        # Actually, let's stick to true complex linear projection: W * z
        # W is complex parameters (Wr, Wi)
        # We model this by having 4 real matrices coupled.
        # For this V2 step, let's keep it computationally simpler:
        # Independent projections for Real and Imag parts, but allow mixing *in attention*.

        kr = self.key_real(x_real)
        ki = self.key_imag(x_imag)
        vr = self.value_real(x_real)
        vi = self.value_imag(x_imag)

        # Reshape for heads
        qr = qr.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        qi = qi.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kr = kr.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        ki = ki.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        vr = vr.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        vi = vi.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Complex Dot Product (Re(Q * K_conj))
        # (Qr + iQi) * (Kr - iKi) = (QrKr + QiKi) + i(QiKr - QrKi)
        # We take the REAL part for the attention score (Energy)
        score_real = torch.matmul(qr, kr.transpose(-2, -1)) + torch.matmul(
            qi, ki.transpose(-2, -1)
        )

        # Scale
        score_real = score_real / math.sqrt(self.head_dim)

        # MASKING
        # 1. Causal Mask (Prevent cheating) - Shape (T, T)
        # We want to mask positions where j > i (future)
        causal_mask = torch.triu(
            torch.ones(T, T, device=score_real.device), diagonal=1
        ).bool()

        # 2. Padding Mask handling
        # mask input is typically (B, 1, 1, T) or similar.
        # We need a combined mask of shape (B, 1, T, T) or (B, H, T, T)

        # Start with 0 (keep)
        extended_mask = torch.zeros(
            B, 1, T, T, device=score_real.device, dtype=torch.bool
        )

        # Apply Causal (Broadcasts to B)
        # causal_mask is (T, T), triggers broadcasting
        extended_mask = extended_mask | causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply Padding (if provided)
        if mask is not None:
            # mask is (B, 1, 1, T) -> 1=valid, 0=pad
            # We want to MASK (set to True) where mask == 0 (pad)
            # Invert mask: (mask == 0) gives True for pads
            pad_mask = mask == 0  # (B, 1, 1, T)
            # Expand to (B, 1, T, T): padding is on KEYS (last dim)
            # Each padded key position should be masked for ALL query positions
            pad_mask = pad_mask.expand(B, 1, T, T)
            extended_mask = extended_mask | pad_mask

        score_real = score_real.masked_fill(extended_mask, float("-inf"))

        attn = F.softmax(score_real, dim=-1)

        # 3. Weighted Sum (Values are Complex)
        # Out = Attn * (Vr + iVi) = (Attn*Vr) + i(Attn*Vi)
        out_r = torch.matmul(attn, vr)
        out_i = torch.matmul(attn, vi)

        # Reshape back
        out_r = out_r.transpose(1, 2).contiguous().view(B, T, C)
        out_i = out_i.transpose(1, 2).contiguous().view(B, T, C)

        # 4. Output Projection
        final_r = self.out_real(out_r)
        final_i = self.out_imag(out_i)

        return final_r, final_i
