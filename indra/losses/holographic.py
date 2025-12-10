import torch
import torch.nn as nn
import torch.nn.functional as F


class HolographicLoss(nn.Module):
    """
    V5 Holographic Loss.
    Minimizes Circular Variance (clustering) of phases and regularizes Magnitude.
    """

    def __init__(self, rho_mag=0.1, rho_phase=0.5, use_teacher_mag=True):
        super().__init__()
        self.rho_mag = rho_mag
        self.rho_phase = rho_phase
        self.use_teacher_mag = use_teacher_mag

    def forward(
        self,
        student_logits,
        student_z,
        teacher_logits=None,
        teacher_z=None,
        targets=None,
    ):
        """
        Args:
            student_logits: [Batch, Seq, Vocab] - Real values for CE
            student_z: [Batch, Seq, Dim] - Complex states
            teacher_logits: (Optional) For distillation
            teacher_z: (Optional)
            targets: [Batch, Seq] - Ground truth indices
        """

        # 1. Main Task (Cross Entropy)
        # Flatten for CE: [Batch*Seq, Vocab]
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)), targets.view(-1)
        )

        # Extract components
        if isinstance(student_z, tuple):
            # Legacy/Manual implementation (r, i)
            r, i = student_z
            mag = torch.sqrt(r**2 + i**2 + 1e-6)
            # Atan2 is stable
            phase = torch.atan2(i, r)
        elif torch.is_complex(student_z):
            # Native complex
            mag = student_z.abs()
            phase = student_z.angle()
        else:
            # Assume real
            mag = student_z.abs()
            phase = torch.zeros_like(student_z)

        # 2. Phase Diversity (Circular Statistics)
        # Calculate Mean Resultant Vector R
        # R = |mean(e^{i * theta})|
        # If clustered, R -> 1. If uniform, R -> 0.
        # We want diversity => Minimize R.
        # We calculate this over the Feature Dimension (last dim).

        complex_phasors = torch.exp(1j * phase)
        # Mean across Feature Dim to check clustering *within* a token's representation
        # OR mean across sequence?
        # "Phase encodes position/context" -> Tokens at different positions should have different phases?
        # But here we likely want the *embedding dimension* to utilize the full circle.
        mean_vector = torch.mean(complex_phasors, dim=-1)
        resultant_length = torch.abs(mean_vector)
        loss_phase = torch.mean(resultant_length)  # Average over Batch/Seq

        # 3. Magnitude Consistency
        loss_mag = torch.tensor(0.0, device=student_logits.device)

        # Simple Soft Constraint: Mag shouldn't explode or vanish too much.
        # Target = 1.0 (Unitary-ish)
        loss_mag = torch.mean((mag - 1.0) ** 2)

        # 4. Total Loss
        total_loss = loss_ce + (self.rho_phase * loss_phase) + (self.rho_mag * loss_mag)

        return total_loss, {
            "ce": loss_ce.item(),
            "phase_clustering": loss_phase.item(),
            "mag_reg": loss_mag.item(),
            "total": total_loss.item(),
        }
