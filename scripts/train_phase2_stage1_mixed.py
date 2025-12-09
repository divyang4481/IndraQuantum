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

# Setup Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/train_stage1_mixed.log",
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
    EPOCHS = 10  # 10 Epochs on mixed data (larger dataset) should be good
    LR = 3e-4  # Standard pre-training LR
    CHECKPOINT_DIR = "checkpoints/phase2_stage1_mixed"
    DATA_PATH = "data/mixed_train.pkl"  # NEW DATASET

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        logging.error(
            f"Data file {DATA_PATH} not found. Run scripts/prepare_mixed_data.py first."
        )
        return

    logging.info("Loading mixed data (WikiText + Alpaca)...")
    with open(DATA_PATH, "rb") as f:
        train_data = pickle.load(f)

    dataset = SimpleDataset(train_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logging.info(f"Loaded {len(train_data)} samples.")

    # 2. Initialize Phase 2 Model
    logging.info(f"Initializing IndraQuantum Phase 2 (Stage 1 Mixed)...")
    vocab_size = 32000
    d_model = 128

    student = IndraQuantumPhase2(vocab_size, d_model).to(device)
    logging.info(f"Student Parameters: {student.get_num_params():,}")

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(student.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Pure CE Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 6. Training Loop
    metrics_file = "logs/training_metrics_stage1_mixed.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "Loss"])

    for epoch in range(EPOCHS):
        student.train()
        total_loss_sum = 0
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()

            # Forward
            s_logits, s_mag, s_phase = student(input_ids)

            # Shift
            s_logits_shift = s_logits[:, :-1, :].contiguous()
            targets_shift = input_ids[:, 1:].contiguous()

            # Mask Padding (ID 2)
            pad_id = 2
            original_targets = targets_shift.clone()
            targets_shift[targets_shift == pad_id] = -100

            # Restore First EOS
            is_pad = original_targets == pad_id
            cum_pad = is_pad.cumsum(dim=1)
            restore_mask = (cum_pad == 1) & is_pad
            targets_shift[restore_mask] = pad_id

            # Compute CE Loss
            loss = criterion(
                s_logits_shift.view(-1, vocab_size), targets_shift.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss.item()

            pbar.set_postfix({"Loss": f"{loss.item():.2f}"})

            if batch_idx % 10 == 0:
                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, batch_idx, loss.item()])

        scheduler.step()

        # Checkpoint
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_stage1_mixed_epoch_{epoch+1}.pt"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "loss": total_loss_sum / len(loader),
            },
            ckpt_path,
        )
        logging.info(f"Saved {ckpt_path}")


if __name__ == "__main__":
    train()
