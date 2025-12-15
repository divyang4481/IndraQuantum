import torch
import torch.nn as nn
from indra.modules.complex_embedding import ComplexEmbedding
from indra.modules.complex_attention import ComplexMultiheadAttention
from indra.modules.complex_ffn import QuantumFeedForward
from indra.modules.complex_norm import ComplexRMSNorm
from indra.modules.complex_linear import ComplexLinear
from indra.modules.complex_dropout import ComplexDropout


class IndraV5Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, window_size=None):
        super().__init__()
        self.attn = ComplexMultiheadAttention(
            d_model, num_heads, dropout=dropout, window_size=window_size
        )
        self.norm1 = ComplexRMSNorm(d_model)
        self.ffn = QuantumFeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = ComplexRMSNorm(d_model)
        self.dropout = ComplexDropout(dropout)  # Apply to complex (careful)

    def forward(self, z, mask=None):
        # Pre-Norm architecture
        residual = z
        z_norm = self.norm1(z)
        attn_out = self.attn(z_norm, z_norm, z_norm, attn_mask=mask)
        z = residual + self.dropout(attn_out)  # Added Dropout

        residual = z
        z_norm = self.norm2(z)
        ffn_out = self.ffn(z_norm)
        z = residual + self.dropout(ffn_out)  # Added Dropout
        return z


from torch.utils.checkpoint import checkpoint


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
        tie_word_embeddings=False,
        gradient_checkpointing=False,  # New Arg
        window_size=None,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 1. Complex Embedding
        self.embedding = ComplexEmbedding(vocab_size, d_model)

        # Additive Positional Embedding
        self.pos_embedding = ComplexEmbedding(max_seq_len, d_model)

        self.dropout_in = ComplexDropout(dropout)

        # 2. Layers
        # 2. Layers
        self.layers = nn.ModuleList(
            [
                IndraV5Block(d_model, num_heads, d_ff, dropout, window_size)
                for _ in range(num_layers)
            ]
        )

        # 3. Output Norm
        self.norm_f = ComplexRMSNorm(d_model)

        # 4. Head
        self.head = ComplexLinear(d_model, vocab_size, bias=False)
        self.final_bias = nn.Parameter(torch.zeros(vocab_size))

        if tie_word_embeddings:
            self.head.weight_real = self.embedding.embed_real.weight
            self.head.weight_imag = self.embedding.embed_imag.weight

    def forward(self, input_ids, mask=None):
        B, L = input_ids.shape

        z = self.embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        pos_z = self.pos_embedding(pos_ids)
        z = z + pos_z
        z = self.dropout_in(z)

        # Create Causal Mask (Required for Decoder)
        # mask shape: [B, 1, L, L] or [L, L] broadcastable
        # attn_mask values: 0 for keep, -inf for mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=z.device) * float("-inf"), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # Combine with padding mask if provided
        # Combine with padding mask if provided
        if mask is not None:
            # mask is [B, L], 1=keep, 0=pad
            # Expand to [B, 1, 1, L] for broadcasting with [B, 1, L, L] causal mask?
            # actually we want [B, 1, 1, L] so that for every query pos, we mask keys that are pads.
            
            # Create additive mask: 0.0 for 1, -inf for 0
            extended_mask = (1.0 - mask[:, None, None, :]) * torch.finfo(z.dtype).min
            
            # causal_mask is [1, 1, L, L] (triu)
            # extended_mask is [B, 1, 1, L]
            
            # We want final mask to be min(causal, padding)
            # causal blocks future. padding blocks pads.
            # causal is -inf for future. padding is -inf for pads.
            # sum works if we use 0 and -inf.
            
            final_mask = causal_mask + extended_mask
        else:
            final_mask = causal_mask

        # Layers with Checkpointing
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # checkpoint requires inputs to require_grad. z usually does.
                # but complex tensors might be tricky. Checkpointing handles it if args are tensors.
                z = checkpoint(layer, z, final_mask, use_reentrant=False)
            else:
                z = layer(z, mask=final_mask)

        z_out = self.norm_f(z)

        z_logits = self.head(z_out)

        mag_sq = z_logits.real.pow(2) + z_logits.imag.pow(2)
        logits = torch.log(mag_sq + 1e-6) + self.final_bias

        mag = z_out.abs()
        phase = z_out.angle()

        return logits, mag, phase
