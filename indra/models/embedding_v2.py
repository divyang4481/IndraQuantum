import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexEmbeddingV2(nn.Module):
    """
    Complex Embedding V2: Explicitly separates Magnitude and Phase.
    z = softplus(M) * e^(i * P)
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Magnitude (Meaning Strength)
        # Init with uniform distribution slightly > 0
        self.raw_mag = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.raw_mag.weight, -0.5, 0.5)

        # Phase (Context/Relationship)
        # Init with uniform distribution -pi to pi
        self.raw_phase = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.raw_phase.weight, -math.pi, math.pi)

    def forward(self, x):
        # 1. Magnitude: Enforce positivity
        # Use softplus to ensure strictly positive magnitude
        mag = F.softplus(self.raw_mag(x)) + 1e-4

        # 2. Phase: Unconstrained geometry
        phase = self.raw_phase(x)

        return mag, phase

    def get_complex_vector(self, x):
        """Returns the actual complex numbers: z = M * (cos P + i sin P)"""
        mag, phase = self(x)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return real, imag
