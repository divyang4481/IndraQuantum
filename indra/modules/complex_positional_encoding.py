import torch
import torch.nn as nn
import math


class ComplexSinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependent on pos and i
        # We want complex phasors: e^(i * theta)
        # theta = pos / 10000^(2i/d_model)

        pe = torch.zeros(max_len, d_model, dtype=torch.cfloat)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 1).float() * (-math.log(10000.0) / d_model)
        )

        # Calculate theta
        theta = position * div_term

        # Create complex phasors: cos(theta) + i*sin(theta)
        # Actually in torch directly: torch.polar(mag, theta)
        # We set Magnitude to 1.0 (Pure Phase) or allow it to be learned?
        # For V6 "Hard Phase", let's use Magnitude 1.0.
        pe = torch.polar(torch.ones_like(theta), theta)

        # Register as buffer (not a learnable parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, SeqLen, Dim] (Complex)
        Returns:
            x + pos_embedding (Additive)
            OR
            x * pos_embedding (Multiplicative/Rotary - V6 Style)
        """
        # We'll use Multiplicative (Rotary-like) for V6 as it couples Phase directly
        # z_pos = z_content * e^(i * pos)
        # This acts as a rotation in the complex plane.

        B, L, D = x.shape
        # Slice pe to current length
        pos_emb = self.pe[:L, :].unsqueeze(0)  # [1, L, D]

        # Multiplicative interaction (Rotation)
        return x * pos_emb
