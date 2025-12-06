"""
Quantum Core Module (Stable Version)
Defines the IndraQuantum model using robust Transformer components
while maintaining the Quantum-Inspired interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class TensorTrainComplexOutput(nn.Module):
    """
    Tensor Train Complex Output Layer.
    Implements W * z and returns |W*z|^2 (Born Rule).
    Factorizes W (out, in) into Complex TT-Cores.
    """

    def __init__(self, in_features, out_features, bias=False, tt_rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # Factorization
        self.in_factors = [4, 4, 8]  # 128
        self.out_factors = [20, 40, 40]  # 32000

        # Real Cores
        self.cores_real = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, self.out_factors[0], self.in_factors[0], tt_rank)
                    * 0.02
                ),
                nn.Parameter(
                    torch.randn(
                        tt_rank, self.out_factors[1], self.in_factors[1], tt_rank
                    )
                    * 0.02
                ),
                nn.Parameter(
                    torch.randn(tt_rank, self.out_factors[2], self.in_factors[2], 1)
                    * 0.02
                ),
            ]
        )

        # Imaginary Cores
        self.cores_imag = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, self.out_factors[0], self.in_factors[0], tt_rank)
                    * 0.02
                ),
                nn.Parameter(
                    torch.randn(
                        tt_rank, self.out_factors[1], self.in_factors[1], tt_rank
                    )
                    * 0.02
                ),
                nn.Parameter(
                    torch.randn(tt_rank, self.out_factors[2], self.in_factors[2], 1)
                    * 0.02
                ),
            ]
        )

        # Bias (Real only, applied to magnitude? Or Complex bias?)
        # Born rule usually implies no bias, but for NN stability we might want one.
        # Let's add a real bias to the final logits.
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _reconstruct_matrix(self, cores):
        # Reconstruct full matrix from cores
        g1, g2, g3 = cores

        # Contract 1 & 2
        temp = torch.einsum("labr,rcdr->labcdr", g1, g2)
        # Contract & 3
        final = torch.einsum("labcdr,refk->labcdefk", temp, g3)

        final = final.squeeze(0).squeeze(-1)
        final = final.permute(0, 2, 4, 1, 3, 5)  # [a, c, e, b, d, f]

        return final.reshape(self.out_features, self.in_features)

    def forward(self, x_real, x_imag=None):
        # x_real: [Batch, Seq, In]
        # x_imag: [Batch, Seq, In] (Optional, defaults to 0)

        if x_imag is None:
            x_imag = torch.zeros_like(x_real)

        # 1. Reconstruct Complex Weight Matrix W = Wr + iWi
        Wr = self._reconstruct_matrix(self.cores_real)
        Wi = self._reconstruct_matrix(self.cores_imag)

        # 2. Complex Multiply: (Wr + iWi) * (xr + ixi)
        # = (Wr*xr - Wi*xi) + i(Wr*xi + Wi*xr)

        real_out = F.linear(x_real, Wr) - F.linear(x_imag, Wi)
        imag_out = F.linear(x_imag, Wr) + F.linear(x_real, Wi)

        # 3. Born Rule: Probability ~ |Amplitude|^2
        # CrossEntropyLoss expects logits (log-probabilities).
        # We compute logits = log(|Amplitude|^2)
        # Add epsilon for numerical stability
        logits = torch.log(real_out**2 + imag_out**2 + 1e-10)

        if self.bias is not None:
            # Add bias to the logits (standard practice)
            logits = logits + self.bias

        return logits


class IndraQuantumLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, config):
        super().__init__()
        # Standard Multihead Attention for Stability
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Standard FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, imag_states=None):
        # x: [Batch, Seq, D] (Real/Fused features)
        # imag_states: [Batch, Seq, D] (Imaginary/Phase features)

        # 1. Causal Mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        # 2. Hopping Bias (Quantum Tunneling)
        # If imag_states are provided, compute interaction: H = I * I^T
        # This represents "Hopping Amplitude" between tokens based on phase
        attn_bias = None
        if imag_states is not None:
            # Project or Normalize? Let's just dot product.
            # Scale by 1/sqrt(D) to match attention scale
            scale = 1.0 / math.sqrt(x.size(-1))
            hopping = torch.bmm(imag_states, imag_states.transpose(1, 2)) * scale

            # hopping is [Batch, Seq, Seq].
            # MultiheadAttention expects attn_mask to be [Seq, Seq] or [Batch*NumHeads, Seq, Seq]
            # We need to broadcast or repeat for heads.
            # Simplest: Add to causal mask? No, causal mask is boolean.
            # We can pass it as attn_mask if we convert causal to float.

            # Combine Causal (Float -inf) and Hopping (Float values)
            # Start with 0
            attn_bias = torch.zeros(
                x.size(0), seq_len, seq_len, device=x.device, dtype=x.dtype
            )
            # Add Hopping
            attn_bias = attn_bias + hopping
            # Apply Causal (Set future to -inf)
            attn_bias.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

            # MultiheadAttention expects [N*NumHeads, L, S] if 3D.
            # We have [N, L, S]. We need to repeat for heads.
            n_heads = self.attn.num_heads
            attn_bias = attn_bias.repeat_interleave(n_heads, dim=0)
        else:
            # Fallback to standard boolean causal mask
            attn_bias = causal_mask

        # Attention
        residual = x
        # Pass attn_bias. If it's a Tensor, it's additive. If Bool, it's masking.
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias, need_weights=False)
        x = self.norm1(residual + self.dropout(attn_out))

        # FFN
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)

        return x


class IndraQuantum(nn.Module):
    """
    IndraQuantum: Optimized Language Model.
    Currently running in STABLE mode (Real-Valued) for baseline performance.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.config = config if config is not None else {}

        # Embeddings
        use_tt = self.config.get("use_tt_embeddings", False)
        tt_rank = self.config.get("tt_rank", 4)

        if use_tt:
            from .embedding import QuantumTTEmbedding

            self.token_embedding = QuantumTTEmbedding(
                vocab_size, d_model, tt_rank=tt_rank
            )
            # Learnable Fusion: Map Complex (Real, Imag) -> Real Feature Space
            # Input: d_model * 2, Output: d_model
            self.complex_fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Layers
        self.layers = nn.ModuleList(
            [
                IndraQuantumLayer(d_model, n_heads, dropout, config=self.config)
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Output Head
        if use_tt:
            self.output_projection = TensorTrainComplexOutput(
                d_model, vocab_size, bias=False, tt_rank=tt_rank
            )
        else:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying (Only possible if not using TT)
            self.output_projection.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)

        imag_states = None
        if self.config.get("use_tt_embeddings", False):
            # QuantumTTEmbedding returns [B, S, D*2] (Real, Imag)
            # Extract Imaginary part for Hopping
            real, imag = torch.chunk(x, 2, dim=-1)
            imag_states = imag

            # Use Learnable Fusion to combine them for the main stream
            x = self.complex_fusion(x)

        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = x + pos_emb
        x = self.dropout(x)

        # Layers
        for layer in self.layers:
            x = layer(x, attention_mask, imag_states=imag_states)

        # Output
        if self.config.get("use_tt_embeddings", False):
            # Pass both Real and Imaginary parts to Complex Output
            logits = self.output_projection(x, imag_states)
        else:
            logits = self.output_projection(x)

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.shape[1] > self.max_seq_length:
                    input_ids = input_ids[:, -self.max_seq_length :]

                logits = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
