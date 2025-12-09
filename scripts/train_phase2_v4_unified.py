import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import glob
import re
import pickle
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import csv
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Components
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.loss_v2 import SemanticDistillationLoss
from scripts.setup_teacher import setup_teacher_model

# Setup Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/train_phase2_v4_unified.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data, logits_dir):
        self.data = data
        self.logits_dir = logits_dir

        # Load Memmaps
        n_samples = len(data)
        # We assume shape (N, 512, 100) - check files or assume based on precompute script
        # To be safe, we try to detect, but we know what we wrote: float16/int16
        # If file doesn't exist, we error out (this class shouldn't be used then)

        val_path = os.path.join(logits_dir, "logits_values.npy")
        ind_path = os.path.join(logits_dir, "logits_indices.npy")

        if os.path.exists(val_path) and os.path.exists(ind_path):
            self.vals = np.memmap(
                val_path, dtype="float16", mode="r", shape=(n_samples, 1024, 100)
            )
            self.inds = np.memmap(
                ind_path, dtype="int16", mode="r", shape=(n_samples, 1024, 100)
            )
            self.has_logits = True
        else:
            logging.error("Precomputed logit files not found!")
            raise FileNotFoundError("Logit files missing")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Get tensors
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.has_logits:
            # Slices from memmap
            # Note: Input data might be shorter than 512 if not padded?
            # But precompute assumed 512.
            # We explicitly slice or pad if needed, but usually we just take what's there.
            # We'll assume the memmap matches the hardcoded 1024 seq_len.
            v = self.vals[idx]
            i = self.inds[idx]

            # If input_ids is smaller than 1024, we should slice?
            # Or assume training data is fixed 1024.
            # Let's slice to match input_ids length to be safe.
            seq_len = input_ids.size(0)
            if seq_len < 1024:
                v = v[:seq_len]
                i = i[:seq_len]

            ret["logit_values"] = torch.from_numpy(
                v.copy()
            )  # copy to avoid negative strides issues sometimes
            ret["logit_indices"] = torch.from_numpy(i.copy()).long()

        return ret


def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_v4_epoch_*.pt"))
    if not checkpoints:
        return None

    # Extract epoch numbers
    def get_epoch(path):
        match = re.search(r"epoch_(\d+).pt", path)
        return int(match.group(1)) if match else 0

    latest_ckpt = max(checkpoints, key=get_epoch)
    return latest_ckpt


def train():
    # Configuration
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 3e-4  # Standard training rate
    CHECKPOINT_DIR = "checkpoints/phase2_v4_unified"
    DATA_PATH = "data/mixed_train.pkl"  # Mixed Data (Wiki + Alpaca)

    # Optimization Config
    USE_PRECOMPUTED = True
    LOGITS_DIR = "data/precomputed_logits_v2"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. Load Data
    logging.info("Loading Mixed Data...")
    if not os.path.exists(DATA_PATH):
        logging.error(
            f"Data file {DATA_PATH} not found. Run scripts/prepare_mixed_data.py first."
        )
        return

    with open(DATA_PATH, "rb") as f:
        train_data = pickle.load(f)

    # Check if we can use precomputed dataset
    use_fast_loader = False
    if USE_PRECOMPUTED and os.path.exists(
        os.path.join(LOGITS_DIR, "logits_values.npy")
    ):
        logging.info("Using Precomputed Logits Dataset (Fast Mode)")
        dataset = PrecomputedDataset(train_data, LOGITS_DIR)
        use_fast_loader = True
    else:
        logging.info("Using Standard Dataset (Live Inference Mode)")
        dataset = SimpleDataset(train_data)

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0
    )
    logging.info(f"Loaded {len(train_data)} samples.")

    # 2. Initialize Student Model (Fresh Start)
    logging.info(f"Initializing IndraQuantum Phase 2 (v4 Unified)...")
    vocab_size = 32000
    d_model = 128
    student = IndraQuantumPhase2(vocab_size, d_model).to(device)

    # 3. Load Teacher (Only if not using precomputed)
    teacher = None
    if not use_fast_loader:
        logging.info("Loading Teacher...")
        teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        teacher = teacher.to(device)
        teacher.eval()
    else:
        logging.info("Teacher skipped (using precomputed logits).")

    # 4. Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed Precision Scaler
    # scaler = torch.cuda.amp.GradScaler() # Deprecated
    scaler = torch.amp.GradScaler("cuda")

    # 5. Check for Resume
    start_epoch = 0
    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt:
        logging.info(f"Resuming from checkpoint: {latest_ckpt}")
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            student.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            start_epoch = checkpoint["epoch"]

            # PROACTIVE FIX: Reset alpha if it drifted negative or too large
            if hasattr(student.output_layer, "alpha"):
                current_alpha = student.output_layer.alpha.item()
                # Reset if too small (<0.0005) or too large (>0.05)
                if current_alpha < 0.0005 or current_alpha > 0.05:
                    logging.info(
                        f"Resetting alpha from {current_alpha} to 0.005 to correct drift."
                    )
                    with torch.no_grad():
                        student.output_layer.alpha.fill_(0.005)

            logging.info(f"Resumed successfully. Starting from Epoch {start_epoch + 1}")
        except Exception as e:
            logging.error(f"Failed to resume from checkpoint: {e}")
            logging.info("Starting from scratch...")

    # 6. Loss Function (Unified)
    criterion = SemanticDistillationLoss(
        lambda_kd=0.1, lambda_phase=0.1, lambda_mag=0.1, temperature=2.0
    ).to(device)

    # 7. Training Loop
    metrics_file = "logs/training_metrics_v4_unified.csv"

    # Prepare metrics file
    file_mode = "a" if start_epoch > 0 and os.path.exists(metrics_file) else "w"
    with open(metrics_file, file_mode, newline="") as f:
        writer = csv.writer(f)
        if file_mode == "w":
            writer.writerow(
                ["Epoch", "Batch", "Total_Loss", "CE", "KD", "Phase", "Mag"]
            )

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}", mininterval=2.0)

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Autocast Context
            # with torch.cuda.amp.autocast(): # Deprecated
            with torch.amp.autocast("cuda"):
                # Forward Student
                s_logits, s_mag, s_phase = student(
                    input_ids, attention_mask=attention_mask
                )

                # Fetch Teacher Logits
                if use_fast_loader:
                    # Reconstruct from Top-K
                    vals = batch["logit_values"].to(device, non_blocking=True).float()
                    inds = batch["logit_indices"].to(device, non_blocking=True)

                    # Create full sparse tensor
                    # Fill with large negative value for ignored positions
                    t_logits = torch.full(
                        (s_logits.size(0), s_logits.size(1), 32000),
                        -100.0,
                        device=device,
                    )
                    t_logits.scatter_(2, inds, vals)
                else:
                    # Forward Teacher (Live)
                    with torch.no_grad():
                        t_out = teacher(input_ids)
                        t_logits = t_out.logits

                # Shift
                s_logits_shift = s_logits[:, :-1, :].contiguous()
                s_mag_shift = s_mag[:, :-1, :].contiguous()
                s_phase_shift = s_phase[:, :-1, :].contiguous()
                t_logits_shift = t_logits[:, :-1, :].contiguous()
                targets_shift = input_ids[:, 1:].clone()

                # Mask Padding Logic
                pad_id = 2
                original_targets = targets_shift.clone()
                targets_shift[targets_shift == pad_id] = -100  # Ignore padding for CE

                is_pad = original_targets == pad_id
                cum_pad = is_pad.cumsum(dim=1)
                restore_mask = (cum_pad == 1) & is_pad
                targets_shift[restore_mask] = pad_id  # Restore first EOS

                # Compute Unified Loss
                loss, terms = criterion(
                    s_logits_shift,
                    s_mag_shift,
                    s_phase_shift,
                    t_logits_shift,  # Teacher logits for KD
                    targets_shift,  # Ground truth for CE
                    current_epoch=epoch,  # For curriculum weighting
                    total_epochs=EPOCHS,
                )

            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # [ANTIGRAVITY FIX] Clamp Alpha to force semantic learning
            if hasattr(student, "output_layer"):
                with torch.no_grad():
                    # Force alpha <= 0.05. softplus(-2.95) ~= 0.05
                    student.output_layer.alpha.data.clamp_(max=-2.95)

            if batch_idx % 10 == 0:
                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch + 1,
                            batch_idx,
                            loss.item(),
                            terms["ce"],
                            terms["kd"],
                            terms["phase"],
                            terms["mag"],
                        ]
                    )

            if batch_idx % 50 == 0:
                step_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    step_ckpt_path,
                )

            pbar.set_postfix(
                {
                    "L": f"{loss.item():.2f}",
                    "CE": f"{terms['ce']:.2f}",
                    "KD": f"{terms['kd']:.2f}",
                }
            )

        scheduler.step()

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_v4_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": loss.item(),
            },
            ckpt_path,
        )
        logging.info(f"Saved {ckpt_path}")


if __name__ == "__main__":
    train()
