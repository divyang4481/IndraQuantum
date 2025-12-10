import torch
import torch.nn as nn
import torch.nn.functional as F
from indra.modules.complex_linear import ComplexLinear
from indra.modules.complex_dropout import ComplexDropout


class QuantumFeedForward(nn.Module):
    """
    V5 Quantum FeedForward with Symplectic Coupling.
    Prevents 'Complex Collapse' by forcing a phase rotation.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = ComplexLinear(d_model, d_ff)
        self.w2 = ComplexLinear(d_ff, d_model)
        self.dropout = ComplexDropout(dropout)

        # Learnable rotation per layer (CRITICAL: prevents collapse)
        # We initialize with small random values
        self.layer_rotation = nn.Parameter(torch.randn(d_model) * 0.1)

    def forward(self, z):
        # 1. First Linear Projection
        h = self.w1(z)

        # 2. Complex Activation (Split Relu/GELU)
        # Apply GELU independently to Real and Imag parts
        h = F.gelu(h.real) + 1j * F.gelu(h.imag)

        # 3. Dropout
        # Naive dropout on complex number (masking both or independent?)
        # PyTorch dropout works on float. We can mask the complex tensor if it supports it,
        # or mask real/imag independently (which changes phase).
        # To preserve phase somewhat, we should drop the whole complex number.
        # But nn.Dropout on complex might strictly zero out elements.
        h = self.dropout(h)

        # 4. Second Linear Projection
        h = self.w2(h)

        # 5. SYMPLECTIC COUPLING / FORCED ROTATION
        # z_out = z * e^{i * theta}
        # This rotation forces the network to maintain phase coherence to pass signal through.

        # Create rotation phasor: e^{i * theta}
        # layer_rotation is shape (d_model,)
        rot_phase = self.layer_rotation
        rot = torch.polar(torch.ones_like(rot_phase), rot_phase)

        # Apply rotation (broadcast over batch/seq dims)
        # h shape: [Batch, Seq, d_model]
        # rot shape: [d_model] -> [1, 1, d_model]
        h = h * rot.unsqueeze(0).unsqueeze(0)

        return h
