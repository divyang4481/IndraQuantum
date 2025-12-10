import torch
import torch.nn as nn
from indra.modules.complex_embedding import ComplexEmbedding
from indra.modules.complex_attention import ComplexMultiheadAttention
from indra.modules.complex_ffn import QuantumFeedForward
from indra.modules.complex_norm import ComplexRMSNorm
from indra.modules.complex_linear import ComplexLinear
from indra.modules.complex_dropout import ComplexDropout


class IndraV5Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = ComplexMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = ComplexRMSNorm(d_model)
        self.ffn = QuantumFeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = ComplexRMSNorm(d_model)
        self.dropout = ComplexDropout(dropout)  # Apply to complex (careful)

    def forward(self, z, mask=None):
        # Pre-Norm architecture
        residual = z
        z_norm = self.norm1(z)
        attn_out = self.attn(z_norm, z_norm, z_norm, attn_mask=mask)
        z = residual + attn_out

        residual = z
        z_norm = self.norm2(z)
        ffn_out = self.ffn(z_norm)
        z = residual + ffn_out
        return z


class IndraV5(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 1. Complex Embedding
        self.embedding = ComplexEmbedding(vocab_size, d_model)
        # Positional Embedding?
        # "Phase encodes position" - per V3/V5 theory.
        # We should add positional phase rotation or just learnable positional embedding?
        # "If we don't strictly enforce Phase usage... Laziness"
        # Let's add standard learnable complex positional embeddings for now.
        self.pos_embedding = ComplexEmbedding(max_seq_len, d_model)

        # 2. Layers
        self.layers = nn.ModuleList(
            [IndraV5Block(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # 3. Output Norm
        self.norm_f = ComplexRMSNorm(d_model)

        # 4. Head (Tie weights with embedding if possible? Complex tying is tricky)
        # Separate Head
        self.head = ComplexLinear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, mask=None):
        B, L = input_ids.shape

        # Embeddings
        z = self.embedding(input_ids)

        # Add Positional Embeddings
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        pos_z = self.pos_embedding(pos_ids)
        z = z + pos_z

        # Layers
        for layer in self.layers:
            z = layer(z, mask=mask)

        z_out = self.norm_f(z)

        # Logits
        # Project to Vocab
        z_logits = self.head(z_out)

        # Convert Complex -> Real for CrossEntropy
        # Use Magnitude: |z|
        logits = z_logits.abs()

        return logits, z_out  # Return z_out for Holographic Loss (Phase analysis)
