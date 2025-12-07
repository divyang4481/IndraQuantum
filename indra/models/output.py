import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Projection for "Meaning" (Real part alignment)
        self.proj_real = nn.Linear(d_model, d_model)

    def forward(self, x_real, x_imag):
        # 1. Prepare Hidden States
        # We assume x_real/x_imag are the output of the Transformer
        h_real = self.proj_real(x_real)
        h_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-6)

        # 2. Get Embedding Weights (Recalculate on fly or access weights)
        # E_real = M * cos(P)
        # E_mag = M

        # Accessing raw weights directly and computing geometry
        # This is expensive if done per batch if vocab is massive, but for 32k it's okay.
        M = F.softplus(self.embedding.raw_mag.weight) + 1e-4
        P = self.embedding.raw_phase.weight
        E_real = M * torch.cos(P)
        # E_imag = M * torch.sin(P) # Not used for output projection currently

        # 3. Channel 1: Real Predictive Match
        # (B, T, D) @ (D, V) -> (B, T, V)
        logits_real = torch.matmul(h_real, E_real.t())

        # 4. Channel 2: Magnitude (Salience) Match
        # (B, T, D) @ (D, V) -> (B, T, V)
        # We want to match "strong" hidden states to "strong" words.
        # But M is positive, so dot product is just magnitude correlation.
        logits_mag = torch.matmul(h_mag, M.t())

        # 5. Combine
        logits = logits_real + (self.alpha * logits_mag)

        return logits
