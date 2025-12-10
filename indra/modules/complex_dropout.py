import torch
import torch.nn as nn


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, z):
        if not self.training or self.p == 0:
            return z

        # Create mask based on Real part shape
        # Mask shape: Same as z
        mask = torch.bernoulli(torch.full_like(z.real, 1 - self.p))

        # Scale
        # Inverted dropout: / (1-p)
        scale = 1.0 / (1 - self.p)

        # Apply mask to complex Z (broadcasts to both real and imag if multiplied as real)
        # z * mask -> (r + ji) * m = rm + j(im)
        # This drops the entire complex number at that position.
        return z * mask * scale
