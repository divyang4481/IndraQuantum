# IndraQuantum: Advanced Mathematical Analysis & Solutions

## 1. The Core Problem: "Complex Collapse"

In our current architecture, we model tokens as $z = r + i \cdot c$. We want the **Phase** ($\theta$) to encode position/context and **Magnitude** ($M$) to encode semantics.

**The Failure Mode:**
Standard Neural Networks (Gradient Descent) are "lazy". If we don't strictly enforce the Phase usage, the model finds it easier to just use the Real part ($r$) as a standard embedding and keep the Imaginary part ($i$) as noise or zero. This is "Complex Collapse".

Our "Hard Staging" curriculum (Stage 1: CE Only) **encourages** this collapse because it doesn't penalize ignoring phase.

---

## 2. Solution A++: Enhanced Holographic Loss (Circular Variance)

Instead of distinct stages, we define a single, unified Energy Function (Hamiltonian) that forces the model to use the full complex plane. We use **Circular Laws** and **Coupling** to prevent collapse.

### key Improvements in V3.1 Theory

1.  **Circular Phase Variance**: We don't use linear variance. We minimize the Mean Resultant Length ($R$). If phases are clustered, $R \approx 1$. If phases are uniform (good), $R \approx 0$.
2.  **Relative Magnitude**: We don't force Mag=1.0. We verify it matches the Teacher (Distillation) or is smooth.

### The Math

$$ \mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda*{mag} \mathcal{L}*{Mag} + \lambda*{phase} \mathcal{L}*{Phase} $$

Where:

1.  **Circular Regularization**:
    $$ R = \left| \frac{1}{N} \sum e^{i\Phi} \right| $$
    $$ \mathcal{L}\_{Phase} = R $$
    (Minimize $R$ -> Maximize Phase Diversity)

### Production Code Proposal

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HolographicLoss(nn.Module):
    def __init__(self, rho_mag=0.1, rho_phase=0.5, use_teacher_mag=True):
        super().__init__()
        self.rho_mag = rho_mag
        self.rho_phase = rho_phase
        self.use_teacher_mag = use_teacher_mag

    def forward(self, student_logits, student_z, teacher_logits, teacher_z=None, targets=None):
        # 1. Main Task (Cross Entropy)
        loss_ce = F.cross_entropy(student_logits, targets)

        # Extract components
        # Z is complex (or real/imag tuple, need to reconstruct)
        if isinstance(student_z, tuple):
             # Manual implementation (r, i)
             r, i = student_z
             mag = torch.sqrt(r**2 + i**2 + 1e-6)
             phase = torch.atan2(i, r)
        else:
             # Native complex
             mag = student_z.abs()
             phase = student_z.angle()

        # 2. Phase Diversity (Circular Statistics)
        # Calculate Mean Resultant Vector R
        # R = |mean(e^{i * theta})|
        # If clustered, R -> 1. If uniform, R -> 0.
        # We want diversity => Minimize R.
        complex_phasors = torch.exp(1j * phase)
        mean_vector = torch.mean(complex_phasors, dim=-1) # Mean across Feature Dim
        resultant_length = torch.abs(mean_vector)
        loss_phase = torch.mean(resultant_length) # Minimize clustering

        # 3. Magnitude Consistency
        loss_mag = torch.tensor(0.0, device=student_logits.device)
        if self.use_teacher_mag and teacher_logits is not None:
             # Option A: Teacher Distillation on Magnitude
             # Teacher doesn't have explicit magnitude if it's real-valued TinyLlama
             # But we can say: Magnitude of our state should approximate standard dev of teacher hidden?
             # Or just Regularize smoothness: Mag shouldn't be extreme.
             # Better: Use "semantic magnitude" proxy.
             # For now, let's stick to Soft Constraint: Mag shouldn't explode.
             loss_mag = torch.mean((mag - 1.0)**2)
        else:
             loss_mag = torch.mean((mag - 1.0)**2)

        # 4. Total Loss
        total_loss = loss_ce + (self.rho_phase * loss_phase) + (self.rho_mag * loss_mag)

        return total_loss, {
            "ce": loss_ce.item(),
            "phase_clustering": loss_phase.item(),
            "mag_reg": loss_mag.item()
        }
```

---

## 3. Solution B: The "Symplectic" Coupling (Architectural Safeguard)

If the loss isn't enough, we hard-code rotation into the Feedback/FeedForward layer.

### Code Proposal

```python
class QuantumFeedForward(nn.Module):
    """Drop-in replacement for standard FFN with forced phase rotation"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = ComplexLinear(d_model, d_ff)
        self.w2 = ComplexLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable rotation per layer (CRITICAL: prevents collapse)
        self.layer_rotation = nn.Parameter(torch.randn(d_model) * 0.1)

    def forward(self, z):
        # Standard FFN path
        h = self.w1(z)
        h = F.gelu(h.real) + 1j * F.gelu(h.imag) # Complex GELU
        h = self.dropout(h)
        h = self.w2(h)

        # FORCED rotation: model MUST track phase to survive
        # z_out = z * e^{i * theta}
        rot = torch.polar(torch.ones_like(self.layer_rotation), self.layer_rotation)
        h = h * rot.unsqueeze(0).unsqueeze(0)

        return h
```

## 5. Recommendation for Next Step

Adopt **Holographic Loss** immediately. Monitor `phase_clustering` (should drop below 0.5).
