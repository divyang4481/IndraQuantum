import torch
import torch.nn as nn
import torch.nn.functional as F
from indra.modules.complex_linear import ComplexLinear


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)
        self.out_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)

        self.dropout_p = dropout

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        # query, key, value are complex tensors [Batch, Seq, Dim]
        B, L, E = query.shape
        H = self.num_heads
        D = self.head_dim

        # Projections
        q = self.q_proj(query).view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(key).view(B, -1, H, D).transpose(1, 2)  # [B, H, S, D]
        v = self.v_proj(value).view(B, -1, H, D).transpose(1, 2)  # [B, H, S, D]

        # Scaled Dot Product Attention: Softmax(Q K* / sqrt(d)) V
        # K* is conjugate
        # Matrix mult: [B, H, L, D] @ [B, H, D, S]
        # k.transpose(-2, -1).conj()

        k_t = k.transpose(-2, -1).conj()
        attn_weights = torch.matmul(q, k_t) / (D**0.5)

        # Attention Mask
        if is_causal:
            # Create causal mask if not provided
            # Logic: mask future tokens
            # But usually passed in attn_mask
            pass  # Simplified for now, rely on attn_mask passed in

        if attn_mask is not None:
            # attn_mask shape broadcastable
            # Usually mask is real (-inf).
            # complex tensor + real mask works?
            # attn_weights is complex.
            # We care about the result of softmax.
            # Pytorch Softmax on complex takes absolute value? No.
            # Softmax on complex is not standard defined for probability.
            # Typically in "Quantum Attention", we take the magnitude of the attention scores?
            # Or we keep it complex?
            # Standard "Complex Valued NN" approach:
            # Often Softmax is applied to the REAL part only (magnitude matches), or |Attn|?
            # If we want probability distribution (sum to 1 real), we usually use Real(QK*).
            # Let's use Real part for Score -> Softmax, then apply to Complex V.
            # "The attention map is distribution of attention" -> Real numbers [0,1].

            # Using Real part of correlation for scores:
            attn_scores = attn_weights.real

            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout_p, training=self.training)

            # Weighted sum of V (Complex)
            # attn_probs: [B, H, L, S] (Real)
            # v: [B, H, S, D] (Complex)
            # Broadcase real probs times complex vector
            attn_probs = torch.complex(
                attn_probs, torch.zeros_like(attn_probs)
            )  # Make complex to matmul
            out = torch.matmul(attn_probs, v)

        else:
            # Fallback simple
            attn_scores = attn_weights.real
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = torch.complex(attn_probs, torch.zeros_like(attn_probs))
            out = torch.matmul(attn_probs, v)

        # [B, H, L, D] -> [B, L, H, D] -> [B, L, E]
        out = out.transpose(1, 2).contiguous().view(B, L, E)

        return self.out_proj(out)
