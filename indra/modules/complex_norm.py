import torch
import torch.nn as nn


class ComplexRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # Real scaling parameter

    def forward(self, z):
        # z: [Batch, Seq, Dim]
        # RMS of magnitude
        # |z|^2 = r^2 + i^2
        norm_sq = z.real.pow(2) + z.imag.pow(2)
        mean_sq = norm_sq.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)

        # Normalize
        z_norm = z / torch.complex(rms, torch.zeros_like(rms))

        # Scale (Real scaling applied to complex vector)
        return z_norm * torch.complex(self.scale, torch.zeros_like(self.scale))
