import torch
import torch.nn as nn
import torch.nn.functional as F
from indra.modules.complex_linear import ComplexLinear


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, window_size=None):
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
        self.window_size = window_size

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

        # AMP Fix: ComplexHalf (fp16 complex) matmul is not supported on CUDA.
        # We must upcast to float32 (complex64) for the dot product.
        q_c64 = q.to(torch.complex64)
        k_t_c64 = k.transpose(-2, -1).conj().to(torch.complex64)

        attn_weights = torch.matmul(q_c64, k_t_c64) / (D**0.5)

        # Attention Mask
        if is_causal:
            # Create causal mask if not provided
            # Logic: mask future tokens
            # But usually passed in attn_mask
            pass  # Simplified for now, rely on attn_mask passed in

        # Window Attention Mask
        if self.window_size is not None:
            # Create banded matrix mask
            # 1 if |i-j| <= w and j <= i (causal) -> actually just j >= i - w
            w = self.window_size
            # L is sequence length
            # mask shape: [L, L]
            # We want to mask where indices are too far apart
            window_mask = torch.triu(
                torch.ones(L, L, device=query.device) * float("-inf"), diagonal=w + 1
            )
            # Also need lower triangular part if not fully causal?
            # But this is specific to causal LMs usually.
            # For causal LM: i is current, j is past.
            # We attend to j if i - w <= j <= i
            # which means j >= i - w.
            # If j < i - w, mask it.
            # triu(1) masks j > i.
            # tril(i-w-1) masks j < i - w.

            # Create a mask that blocks everything OUTSIDE the window
            # We assume standard Causal is handled by 'attn_mask' or 'is_causal' (which we passed)
            # The window mask just adds "too far past" masking.

            # Mask indices where j < i - w
            # This is equivalent to lower triangular with offset -w-1

            lower_mask = torch.tril(
                torch.ones(L, L, device=query.device) * float("-inf"), diagonal=-(w + 1)
            )

            if attn_mask is None:
                attn_mask = lower_mask
            else:
                attn_mask = attn_mask + lower_mask

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

            # AMP Fix 2: Attn @ V also fails in ComplexHalf.
            # Cast both to complex64 for matmul.

            attn_probs_c = torch.complex(attn_probs, torch.zeros_like(attn_probs)).to(
                torch.complex64
            )

            v_c64 = v.to(torch.complex64)

            out = torch.matmul(attn_probs_c, v_c64)

        else:
            # Fallback simple
            attn_scores = attn_weights.real
            attn_probs = F.softmax(attn_scores, dim=-1)

            attn_probs_c = torch.complex(attn_probs, torch.zeros_like(attn_probs)).to(
                torch.complex64
            )
            v_c64 = v.to(torch.complex64)

            out = torch.matmul(attn_probs_c, v_c64)

        # [B, H, L, D] -> [B, L, H, D] -> [B, L, E]
        # Cast back to original dtype to continue
        out = out.to(query.dtype)
        out = out.transpose(1, 2).contiguous().view(B, L, E)

        return self.out_proj(out)
