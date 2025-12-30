import torch
import torch.nn as nn


class ComplexRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None):
        """
        RoPE for Complex Numbers is elegantly simple.
        Standard RoPE rotates a pair (x,y) by theta.
        Complex Numbers ARE pairs (x,iy).
        So Complex RoPE is just multiplication by e^(i*theta).

        z_rotated = z * e^(i*theta)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate Theta for each dimension pair
        # dim must be divisible by 2 for standard RoPE, but here 'dim' is actually the complex dim.
        # So we have 'dim' unique angles.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))

        # We need coverage for full complex dim. Standard RoPE pairs features.
        # In complex space, each feature is already a 'pair' (Real, Imag).
        # So we generate an angle for EACH of the d_model dimensions?
        # Standard RoPE: [x1, x2, x3, x4] -> Rotate (x1,x2) and (x3,x4).
        # Complex: [z1, z2] -> Rotate z1, Rotate z2.
        # So we need 'dim' freqencies, not dim/2.
        # Wait, if we use standard logic, we want relative distance encoding.

        # Let's stick to the standard frequency calculation but apply it per complex feature.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 1).float().to(device) / dim))

        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device="cpu", dtype=torch.float32):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # freqs: [seq_len, dim]
        freqs = torch.outer(t, self.inv_freq)

        # Create Complex Rotation Factor: e^(i*theta) = cos(theta) + i*sin(theta)
        # shape: [seq_len, dim]
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)

    def forward(self, z, seq_len=None):
        """
        z: Complex Tensor [Batch, SeqLen, Heads, Dim] or [Batch, SeqLen, Dim]
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=z.device, dtype=z.real.dtype)

        # Get cached cos/sin for this sequence length
        # [SeqLen, Dim]
        cos = self.cos_cached[:seq_len, ...].to(z.device)
        sin = self.sin_cached[:seq_len, ...].to(z.device)

        # Proper broadcasting
        # z is typically [B, L, H, D]
        # We need to unsqueeze for Batch and Head dims
        # cos becomes [1, L, 1, D]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Complex Multiplication:
        # z_new = z * (cos + i*sin)
        #       = (z.r + i*z.i) * (cos + i*sin)
        #       = (z.r*cos - z.i*sin) + i(z.r*sin + z.i*cos)

        t_real = z.real * cos - z.imag * sin
        t_imag = z.real * sin + z.imag * cos

        return torch.complex(t_real, t_imag)
