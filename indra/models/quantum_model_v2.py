import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import Phase 2 Components
from .embedding_v2 import ComplexEmbeddingV2
from .attention import ComplexAttention
from .output import HybridOutput


class IndraQuantumPhase2(nn.Module):
    """
    IndraQuantum Phase 2: FULL COMPLEX ARCHITECTURE
    - Complex Embeddings (Mag+Phase)
    - Complex Attention
    - Hybrid Output
    """

    def __init__(self, vocab_size, d_model, n_layers=4, n_heads=4, config=None):
        super().__init__()
        self.config = config if config else {}
        self.d_model = d_model

        self.mode = self.config.get("mode", "fullq")  # Default to full quantum

        if self.mode == "real":
            # Real Baseline: Standard Transformer
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(2048, d_model)  # Learned Positional
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        batch_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.output_layer = nn.Linear(d_model, vocab_size)

        elif self.mode == "complex_embed":
            # Complex Embedding -> Real Attention
            self.token_embedding = ComplexEmbeddingV2(vocab_size, d_model)
            self.register_buffer(
                "inv_freq",
                1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)),
            )
            # Projection Complex -> Real (e.g. Real part + Imag part projection)
            self.complex_to_real = nn.Linear(d_model * 2, d_model)

            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        batch_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.output_layer = nn.Linear(d_model, vocab_size)

        else:  # fullq
            # 1. Complex Embedding V2
            self.token_embedding = ComplexEmbeddingV2(vocab_size, d_model)

            # Positional Embeddings (Absolute)
            self.pos_embed_real = nn.Embedding(2048, d_model)
            self.pos_embed_imag = nn.Embedding(2048, d_model)

            # 2. Complex Transformer Layers
            self.layers = nn.ModuleList(
                [ComplexLayer(d_model, n_heads) for _ in range(n_layers)]
            )

            # 3. Hybrid Output
            self.output_layer = HybridOutput(d_model, vocab_size, self.token_embedding)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape

        if self.mode == "real":
            x = self.token_embedding(input_ids)
            # Add Positional
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
            x = x + self.pos_embedding(pos)

            # Mask for PyTorch Transformer expects (B, T) boolean mask where True is IGNORED?
            # Or attn_mask (T, T) or key_padding_mask (B, T).
            # We pass src_key_padding_mask = ~attention_mask (if mask is 1 for valid)
            key_padding_mask = None
            if attention_mask is not None:
                # attention_mask: 1=valid, 0=pad. PyTorch expects True=pad.
                key_padding_mask = attention_mask == 0

            for layer in self.layers:
                x = layer(x, src_key_padding_mask=key_padding_mask)

            logits = self.output_layer(x)

            # Dummy Quantum Stats for Real Mode
            final_mag = torch.abs(x)  # Just use absolute value
            final_phase = torch.zeros_like(x)  # No phase

            return logits, final_mag, final_phase

        elif self.mode == "complex_embed":
            # Complex Embed
            mag, phase = self.token_embedding(input_ids)

            # Phase Rotation (Positional)
            pos = torch.arange(T, device=input_ids.device, dtype=self.inv_freq.dtype)
            sinusoid_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
            pos_angles = torch.cat((sinusoid_inp, sinusoid_inp), dim=-1)
            phase = phase + pos_angles.unsqueeze(0)

            # Convert to Real Vector
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            # Concatenate or Project?
            # We defined complex_to_real as Linear(2D -> D)
            x_complex = torch.cat([real, imag], dim=-1)
            x = self.complex_to_real(x_complex)

            # Standard Transformer Process
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0

            for layer in self.layers:
                x = layer(x, src_key_padding_mask=key_padding_mask)

            logits = self.output_layer(x)

            # Reconstruct Pseudo-Quantum Stats
            final_mag = torch.abs(x)
            final_phase = torch.atan2(imag, real)  # From input (approximation) or zeros

            return logits, final_mag, final_phase

        else:  # fullq (Original Logic)
            # Embed (Mag, Phase)
            mag, phase = self.token_embedding(input_ids)

            # Additive Positional Encoding (Standard but applied to Complex)
            # We use absolute positions for stability in this Tiny implementation.
            # pos_real/pos_imag: [1, T, D]
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

            # Convert to Cartesian for Linear Processing
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)

            # Add Position Information
            real = real + self.pos_embed_real(pos)
            imag = imag + self.pos_embed_imag(pos)

            # Process Layers
            # We need to reshape mask for attention: (B, 1, 1, T) or (B, 1, T, T) for causal?
            # The complex attention expects (B, 1, 1, T) usually for padding mask.
            if attention_mask is not None:
                # Mask is (B, T) with 1 for token, 0 for pad
                # Expand to (B, 1, 1, T) for broadcasting over heads and query length
                mask = attention_mask.view(B, 1, 1, T)
            else:
                mask = None

            for layer in self.layers:
                real, imag = layer(real, imag, mask=mask)

            # Output
            logits = self.output_layer(real, imag)

            # Return final mag/phase from transformed hidden state, not embedding
            # This is what the loss function should supervise
            final_mag = torch.sqrt(real**2 + imag**2 + 1e-6)
            final_phase = torch.atan2(imag, real)

            return logits, final_mag, final_phase

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class ComplexLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = ComplexAttention(d_model, n_heads)
        self.norm1_real = nn.LayerNorm(d_model)
        self.norm1_imag = nn.LayerNorm(d_model)

        # Split FFN? Or Single FFN on Real/Imag independently?
        # Let's do Real FFN + Imag FFN
        self.ffn_real = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ffn_imag = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

        self.norm2_real = nn.LayerNorm(d_model)
        self.norm2_imag = nn.LayerNorm(d_model)

    def forward(self, r, i, mask=None):
        # Attention
        ar, ai = self.attn(r, i, mask=mask)
        r = self.norm1_real(r + ar)
        i = self.norm1_imag(i + ai)

        # FFN
        fr = self.ffn_real(r)
        fi = self.ffn_imag(i)

        r = self.norm2_real(r + fr)
        i = self.norm2_imag(i + fi)

        return r, i
