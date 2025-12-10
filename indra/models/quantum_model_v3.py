import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers_v3 import ComplexLinear, ComplexLayerNorm, ComplexSplitGELU


class QuantumRotaryEmbedding(nn.Module):
    """
    Quantum RoPE: Unitary Rotation e^{i * theta}
    Manual Implementation for Real/Imag parts.
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        # Frequencies for Phase Rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x_real, x_imag, seq_len):
        # x shape: [Batch, SeqLen, Heads, HeadDim]
        t = torch.arange(seq_len, device=x_real.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # [T, D/2] -> [T, D]
        emb = torch.cat((freqs, freqs), dim=-1).to(x_real.device)  # [T, D]
        emb = emb.view(1, seq_len, 1, -1)

        sin_emb = emb.sin()
        cos_emb = emb.cos()

        # Rotation: (r + qi)(c + is) = (rc - qs) + i(rs + qc)
        new_real = x_real * cos_emb - x_imag * sin_emb
        new_imag = x_real * sin_emb + x_imag * cos_emb

        return new_real, new_imag


class ComplexAttentionV3(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0

        self.q_proj = ComplexLinear(d_model, d_model, bias=False)
        self.k_proj = ComplexLinear(d_model, d_model, bias=False)
        self.v_proj = ComplexLinear(d_model, d_model, bias=False)
        self.o_proj = ComplexLinear(d_model, d_model, bias=False)

        self.rotary = QuantumRotaryEmbedding(self.head_dim)

    def forward(self, x_real, x_imag, mask=None):
        B, T, D = x_real.shape
        H = self.num_heads
        HD = self.head_dim

        # 1. Projections
        qr, qi = self.q_proj(x_real, x_imag)
        kr, ki = self.k_proj(x_real, x_imag)
        vr, vi = self.v_proj(x_real, x_imag)

        # 2. Reshape [B, T, H, HD]
        qr = qr.view(B, T, H, HD)
        qi = qi.view(B, T, H, HD)
        kr = kr.view(B, T, H, HD)
        ki = ki.view(B, T, H, HD)
        vr = vr.view(B, T, H, HD)
        vi = vi.view(B, T, H, HD)

        # 3. Apply Rotary (Phase Rotation)
        qr, qi = self.rotary(qr, qi, T)
        kr, ki = self.rotary(kr, ki, T)

        # 4. Transpose for Attention [B, H, T, HD]
        qr, qi = qr.transpose(1, 2), qi.transpose(1, 2)
        kr, ki = kr.transpose(1, 2), ki.transpose(1, 2)
        vr, vi = vr.transpose(1, 2), vi.transpose(1, 2)

        # 5. Score = Re(Q * K_conj)
        # (Qr + iQi)(Kr - iKi) = (QrKr + QiKi) + i(...)
        # We only want Real part.
        scores = torch.matmul(qr, kr.transpose(-1, -2)) + torch.matmul(
            qi, ki.transpose(-1, -2)
        )
        scores = scores / math.sqrt(HD)

        if mask is not None:
            # Mask is True for Invalid/Pad positions
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        # 6. Apply to V
        # Out = Attnw * (Vr + iVi)
        out_r = torch.matmul(attn_weights, vr)
        out_i = torch.matmul(attn_weights, vi)

        # 7. Reshape Back
        out_r = out_r.transpose(1, 2).contiguous().view(B, T, D)
        out_i = out_i.transpose(1, 2).contiguous().view(B, T, D)

        # 8. Output Project
        final_r, final_i = self.o_proj(out_r, out_i)

        return final_r, final_i


class ComplexTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = ComplexAttentionV3(d_model, num_heads)
        self.norm1 = ComplexLayerNorm(d_model)

        self.ffn = nn.Sequential(
            ComplexLinear(d_model, d_model * 4),
            ComplexSplitGELU(),
            ComplexLinear(d_model * 4, d_model),
        )
        self.norm2 = ComplexLayerNorm(d_model)

    def forward(self, r, i, mask=None):
        # Attn Block
        ar, ai = self.attn(r, i, mask)
        r = r + ar
        i = i + ai
        r, i = self.norm1(r, i)

        # FFN Block
        fr, fi = self.ffn[0](r, i)
        fr, fi = self.ffn[1](fr, fi)
        fr, fi = self.ffn[2](fr, fi)

        r = r + fr
        i = i + fi
        r, i = self.norm2(r, i)

        return r, i


class IndraQuantumPhase3(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.emb_mag = nn.Embedding(vocab_size, d_model)
        self.emb_phase = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [ComplexTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        self.output_proj = ComplexLinear(d_model, vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.emb_mag.weight, mean=1.0, std=0.02)
        nn.init.uniform_(self.emb_phase.weight, -math.pi, math.pi)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape

        mag = F.softplus(self.emb_mag(input_ids))
        phase = self.emb_phase(input_ids)

        r = mag * torch.cos(phase)
        i = mag * torch.sin(phase)

        # Mask: 1/True for Invalid, 0/False for Valid
        causal_mask = torch.triu(
            torch.ones(T, T, device=r.device), diagonal=1
        ).bool()  # Upper Tri is invalid
        mask = torch.zeros(B, 1, T, T, device=r.device).bool()
        mask = mask | causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # attention_mask: 1=valid, 0=pad.
            # We want True where invalid.
            pad_mask = (attention_mask == 0).view(B, 1, 1, T)
            mask = mask | pad_mask

        for layer in self.layers:
            r, i = layer(r, i, mask=mask)

        # Measurement
        out_r, out_i = self.output_proj(r, i)

        # Logits = Re(Out) + Bias
        logits = out_r + self.output_bias

        final_mag = torch.sqrt(r**2 + i**2 + 1e-6)
        final_phase = torch.atan2(i, r)

        return logits, final_mag, final_phase
