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
        # Use reduction='none' to handle masking manually
        self.mse = nn.MSELoss(reduction="none")
        self.kl = nn.KLDivLoss(reduction="none")

    def normalize_entropy(self, log_probs):
        """
        Calculates normalized entropy (0 to 1) from log_probs.
        H = -sum(p * log_p)
        Max H = log(V)
        """
        p = torch.exp(log_probs)
        entropy = -torch.sum(p * log_probs, dim=-1)
        max_entropy = math.log(log_probs.size(-1))
        # Clamp for numerical stability
        entropy = torch.clamp(entropy, min=1e-8, max=max_entropy)
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

        # Create Mask for Padding (targets are -100 for padding)
        # Flatten everything for easier masking
        batch_size, seq_len, vocab_size = student_logits.shape
        flat_targets = targets.view(-1)
        mask = (flat_targets != -100).float()
        num_valid = mask.sum()

        if num_valid == 0:
            return torch.tensor(
                0.0, device=student_logits.device, requires_grad=True
            ), {
                "ce": 0.0,
                "kd": 0.0,
                "phase": 0.0,
                "mag": 0.0,
                "kd_w": 0.0,
                "aux_w": 0.0,
            }

        # 1. Predictive Loss (Cross-Entropy)
        # CrossEntropyLoss already handles ignore_index=-100
        loss_ce = self.ce(student_logits.view(-1, vocab_size), flat_targets)

        # 2. Adaptive Weights (Curriculum Learning)
        progress = current_epoch / total_epochs

        # KD: 0.01 -> 0.05 (Strictly clamped)
        kd_weight = 0.01 + (progress * 0.04)
        kd_weight = min(0.05, max(0.01, kd_weight))

        # Aux: 0.0 -> 0.1
        # Starts ramping at 20% progress. Max value is (1.0 - 0.2) * 0.125 = 0.1
        aux_weight = max(0.0, (progress - 0.2) * 0.125) if progress > 0.2 else 0.0

        # Teacher Probs
        T = self.temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)

        # 3. Imitative Loss (KD)
        # Calculate KL for all, then mask
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL reduction='none' returns (B, T, V). Sum over V (dim=-1) to get token-level KL
        kl_per_token = self.kl(s_log_probs, teacher_probs).sum(dim=-1)  # (B, T)
        loss_kd = (kl_per_token.view(-1) * mask).sum() / num_valid * (T * T)

        # 4. Certainty Loss (Phase)
        with torch.no_grad():
            norm_H = self.normalize_entropy(teacher_log_probs)  # (B, T)
            target_phase = norm_H * math.pi

        student_avg_phase = torch.mean(torch.abs(student_phase), dim=-1)  # (B, T)
        loss_phase_raw = self.mse(student_avg_phase, target_phase)  # (B, T)
        loss_phase = (loss_phase_raw.view(-1) * mask).sum() / num_valid

        # 5. Salience Loss (Magnitude)
        with torch.no_grad():
            max_prob, _ = torch.max(teacher_probs, dim=-1)  # (B, T)
            target_mag = max_prob * 5.0

        student_avg_mag = torch.mean(student_mag, dim=-1)  # (B, T)
        loss_mag_raw = self.mse(student_avg_mag, target_mag)  # (B, T)
        loss_mag = (loss_mag_raw.view(-1) * mask).sum() / num_valid

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
