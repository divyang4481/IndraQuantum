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

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Phase 2 Components
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.loss_v2 import SemanticDistillationLoss
from scripts.setup_teacher import setup_teacher_model

# Setup Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/train_phase2_stage2.log",
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


def train():
    # Configuration
    BATCH_SIZE = 8
    EPOCHS = 20  # Phase 2 v3 run
    LR = 1e-4  # Gentle learning rate
    CHECKPOINT_DIR = "checkpoints/phase2_stage2_v3"  # NEW DIR for v3
    PRETRAINED_PATH = "checkpoints/phase2_stage1/checkpoint_stage1_epoch_2.pt"
    DATA_PATH = "data/wikitext/train.pkl"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load Data
    logging.info("Loading data...")
    with open(DATA_PATH, "rb") as f:
        train_data = pickle.load(f)
    dataset = SimpleDataset(train_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    logging.info(f"Initializing IndraQuantum Phase 2 (Stage 2 v3)...")
    vocab_size = 32000
    d_model = 128
    student = IndraQuantumPhase2(vocab_size, d_model).to(device)

    # Resume Logic
    start_epoch = 0
    existing_checkpoints = glob.glob(
        os.path.join(CHECKPOINT_DIR, "checkpoint_stage2_epoch_*.pt")
    )

    if existing_checkpoints:

        def get_epoch(path):
            try:
                return int(re.search(r"epoch_(\d+).pt", path).group(1))
            except:
                return -1

        latest_ckpt = max(existing_checkpoints, key=get_epoch)
        start_epoch = get_epoch(latest_ckpt)

        logging.info(f"Resuming from {latest_ckpt} (Epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
    else:
        # Load Stage 1 Weights
        if os.path.exists(PRETRAINED_PATH):
            logging.info(f"Loading Stage 1 weights from {PRETRAINED_PATH}...")
            ckpt = torch.load(PRETRAINED_PATH, map_location=device)
            student.load_state_dict(ckpt["model_state_dict"])
        else:
            logging.error("Stage 1 Checkpoint not found! Aborting.")
            return

    # Load Teacher
    logging.info("Loading Teacher...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher = teacher.to(device)
    teacher.eval()

    # Optimizer & Loss
    optimizer = optim.AdamW(student.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Semantic Distillation Loss
    # v3 Strategy: Gentle Distillation (Max KD weight 0.05)
    criterion = SemanticDistillationLoss(
        lambda_kd=0.1, lambda_phase=0.1, lambda_mag=0.1, temperature=2.0
    ).to(device)

    # Training Loop
    metrics_file = "logs/training_metrics_stage2_v3.csv"  # New log file v3
    mode = "a" if start_epoch > 0 else "w"
    with open(metrics_file, mode, newline="") as f:
        writer = csv.writer(f)
        if start_epoch == 0:
            writer.writerow(
                ["Epoch", "Batch", "Total_Loss", "CE", "KD", "Phase", "Mag"]
            )

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()

            # Forward Student
            s_logits, s_mag, s_phase = student(input_ids)

            # Forward Teacher
            with torch.no_grad():
                t_out = teacher(input_ids)
                t_logits = t_out.logits

            # Shift
            s_logits_shift = s_logits[:, :-1, :].contiguous()
            s_mag_shift = s_mag[:, :-1, :].contiguous()
            s_phase_shift = s_phase[:, :-1, :].contiguous()
            t_logits_shift = t_logits[:, :-1, :].contiguous()
            targets_shift = input_ids[:, 1:].contiguous()

            # Mask Padding Logic (First EOS Restore)
            pad_id = 2
            original_targets = targets_shift.clone()
            targets_shift[targets_shift == pad_id] = -100

            is_pad = original_targets == pad_id
            cum_pad = is_pad.cumsum(dim=1)
            restore_mask = (cum_pad == 1) & is_pad
            targets_shift[restore_mask] = pad_id

            # Compute Loss
            loss, terms = criterion(
                s_logits_shift,
                s_mag_shift,
                s_phase_shift,
                t_logits_shift,
                targets_shift,
                current_epoch=epoch,
                total_epochs=EPOCHS,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

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

            pbar.set_postfix(
                {
                    "L": f"{loss.item():.2f}",
                    "CE": f"{terms['ce']:.2f}",
                    "KD": f"{terms['kd']:.2f}",
                }
            )

        scheduler.step()

        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_stage2_epoch_{epoch+1}.pt"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "loss": loss.item(),
            },
            ckpt_path,
        )
        logging.info(f"Saved {ckpt_path}")


if __name__ == "__main__":
    train()
