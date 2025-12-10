import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@torch.jit.script
def _compute_geometry(
    raw_mag: torch.Tensor, raw_phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M = F.softplus(raw_mag) + 1e-4
    E_real = M * torch.cos(raw_phase)
    return M, E_real


class HybridOutput(nn.Module):
    """
    Hybrid Output Layer:
    Logits = (H_real * E_real^T) + alpha * (H_mag * E_mag^T)
    """

    def __init__(self, d_model, vocab_size, embedding_layer):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = embedding_layer  # Reference to ComplexEmbeddingV2

        # Trainable mixing weight (alpha)
        # Frozen as per Antigravity suggestion for stability
        # Initialize to -5.0 so softplus(-5.0) is approx 0.006 (starts low)
        self.alpha = nn.Parameter(torch.tensor(-5.0), requires_grad=False)

        # Projection for "Meaning" (Real part alignment)
        self.proj_real = nn.Linear(d_model, d_model)

        # Output Bias (Critical for unigram frequency learning)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x_real, x_imag):
        # 1. Prepare Hidden States
        # We assume x_real/x_imag are the output of the Transformer
        h_real = self.proj_real(x_real)
        h_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-6)

        # 2. Get Embedding Weights (Recalculate on fly or access weights)
        # E_real = M * cos(P)
        # E_mag = M

        # Accessing raw weights directly and computing geometry
        # Optimized with JIT to fuse kernels
        #   M = F.softplus(self.embedding.raw_mag.weight) + 1e-4
        # P = self.embedding.raw_phase.weight
        # E_real = M * torch.cos(P)
        M, E_real = _compute_geometry(
            self.embedding.raw_mag.weight, self.embedding.raw_phase.weight
        )
        # E_imag = M * torch.sin(P) # Not used for output projection currently

        # 3. Channel 1: Real Predictive Match
        # (B, T, D) @ (D, V) -> (B, T, V)
        # ANTIGRAVITY FIX: Scaled Dot Product attention (1/sqrt(d))
        logits_real = torch.matmul(h_real, E_real.t()) / math.sqrt(self.d_model)

        # 4. Channel 2: Magnitude (Salience) Match
        # (B, T, D) @ (D, V) -> (B, T, V)
        # We want to match "strong" hidden states to "strong" words.
        # But M is positive, so dot product is just magnitude correlation.
        # ANTIGRAVITY FIX: Scaled Dot Product
        logits_mag = torch.matmul(h_mag, M.t()) / math.sqrt(self.d_model)

        # 5. Combine
        # Use softplus to enforce positivity while maintaining smooth gradients
        # abs() has zero gradient at 0 and flips sign, causing instability
        alpha_positive = F.softplus(self.alpha) + 1e-4
        logits = logits_real + (alpha_positive * logits_mag)

        # Add Bias
        logits = logits + self.bias

        return logits
