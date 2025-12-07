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

        # 1. Complex Embedding V2
        self.token_embedding = ComplexEmbeddingV2(vocab_size, d_model)
        self.position_embedding = nn.Embedding(
            512, d_model
        )  # Standard Positional for now

        # 2. Complex Transformer Layers
        self.layers = nn.ModuleList(
            [ComplexLayer(d_model, n_heads) for _ in range(n_layers)]
        )

        # 3. Hybrid Output
        self.output_layer = HybridOutput(d_model, vocab_size, self.token_embedding)

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Embed
        mag, phase = self.token_embedding(input_ids)

        # Add Positional (Apply to Mag or Phase? Add to Mag, Rotate Phase?)
        # Simple: Add to Mag
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)

        # To make pos_emb complex, let's assume it rotates phase slightly?
        # NO, keep it simple. Add to Real part of "Mag" interpretation?
        # Actually, let's treat "Mag" as the semantic vector magnitude,
        # but we need a Real/Imag vector for processing.

        # Convert to Cartesian for Linear Processing
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        real = real + pos_emb  # Add pos info to real component

        # Process Layers
        for layer in self.layers:
            real, imag = layer(real, imag)

        # Output
        logits = self.output_layer(real, imag)

        # Return everything needed for loss
        return logits, mag, phase

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

    def forward(self, r, i):
        # Attention
        ar, ai = self.attn(r, i)
        r = self.norm1_real(r + ar)
        i = self.norm1_imag(i + ai)

        # FFN
        fr = self.ffn_real(r)
        fi = self.ffn_imag(i)

        r = self.norm2_real(r + fr)
        i = self.norm2_imag(i + fi)

        return r, i
