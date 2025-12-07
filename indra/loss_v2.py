import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SemanticDistillationLoss(nn.Module):
    def __init__(
        self, lambda_kd=0.1, lambda_phase=0.1, lambda_mag=0.1, temperature=2.0
    ):
        super().__init__()
        self.lambda_kd = lambda_kd
        self.lambda_phase = lambda_phase
        self.lambda_mag = lambda_mag
        self.temperature = temperature

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def normalize_entropy(self, log_probs):
        """
        Calculates normalized entropy (0 to 1) from log_probs.
        H = -sum(p * log_p)
        Max H = log(V)
        """
        p = torch.exp(log_probs)
        entropy = -torch.sum(p * log_probs, dim=-1)
        max_entropy = math.log(log_probs.size(-1))
        return entropy / max_entropy

    def forward(
        self,
        student_logits,
        student_mag,
        student_phase,
        teacher_logits,
        targets,
        current_epoch=0,
        total_epochs=50,
    ):
        # Ensure teacher logits are float32 for stability in loss calculation
        teacher_logits = teacher_logits.float()

        # 1. Predictive Loss (Cross-Entropy)
        loss_ce = self.ce(
            student_logits.view(-1, student_logits.size(-1)), targets.view(-1)
        )

        # 2. Adaptive Weights (Curriculum Learning)
        # Ramp up KD/Phase/Mag influence over time
        progress = current_epoch / total_epochs
        # KD: 0.01 -> 0.2
        kd_weight = 0.01 + (progress * 0.19)
        # Aux: 0.0 -> 0.1 (Don't confuse early training)
        aux_weight = max(0.0, (progress - 0.2) * 0.125) if progress > 0.2 else 0.0

        # 3. Imitative Loss (KD)
        T = self.temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)

        loss_kd = self.kl(F.log_softmax(student_logits / T, dim=-1), teacher_probs) * (
            T * T
        )

        # 4. Certainty Loss (Phase)
        # Map Teacher Entropy -> Target Phase (0 to pi)
        # Low Entropy (Confident) -> Phase 0
        # High Entropy (Confused) -> Phase pi
        with torch.no_grad():
            norm_H = self.normalize_entropy(teacher_log_probs)  # (B, T)
            target_phase = norm_H * math.pi

        # Student phase is (B, T, D) -> Take mean over dim to get "Token Phase"
        # Or should we match distribution? Mean is safer for now.
        student_avg_phase = torch.mean(torch.abs(student_phase), dim=-1)  # (B, T)
        loss_phase = self.mse(student_avg_phase, target_phase)

        # 5. Salience Loss (Magnitude)
        # Map Teacher Max Prob (Confidence) -> Target Magnitude
        # High Conf -> High Mag
        with torch.no_grad():
            max_prob, _ = torch.max(teacher_probs, dim=-1)  # (B, T)
            target_mag = max_prob * 5.0  # Scale to reasonable embedding mag range

        student_avg_mag = torch.mean(student_mag, dim=-1)  # (B, T)
        loss_mag = self.mse(student_avg_mag, target_mag)

        # Total
        total_loss = (
            loss_ce
            + (kd_weight * loss_kd)
            + (aux_weight * loss_phase)
            + (aux_weight * loss_mag)
        )

        return total_loss, {
            "ce": loss_ce.item(),
            "kd": loss_kd.item(),
            "phase": loss_phase.item(),
            "mag": loss_mag.item(),
            "kd_w": kd_weight,
            "aux_w": aux_weight,
        }
