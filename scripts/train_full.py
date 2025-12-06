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

from indra.models.quantum_core import IndraQuantum
from scripts.setup_teacher import setup_teacher_model

# Setup Logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/train_full.log",
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


def get_latest_checkpoint(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not files:
        return None, 0

    # Extract epoch numbers
    epochs = []
    for f in files:
        match = re.search(r"checkpoint_epoch_(\d+).pt", f)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        return None, 0

    max_epoch = max(epochs)
    latest_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{max_epoch}.pt")
    return latest_file, max_epoch


def train():
    # Configuration
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 5e-4
    TT_RANK = 64
    CHECKPOINT_DIR = "checkpoints/full_training_rank64"
    DATA_PATH = "data/wikitext/train.pkl"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. Load Data
    logging.info("Loading data...")
    if not os.path.exists(DATA_PATH):
        logging.error(
            f"Data not found at {DATA_PATH}. Please run scripts/prepare_data.py."
        )
        return

    with open(DATA_PATH, "rb") as f:
        train_data = pickle.load(f)

    dataset = SimpleDataset(train_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logging.info(f"Loaded {len(train_data)} samples.")

    # 2. Initialize Model
    logging.info(f"Initializing IndraQuantum (TT-Rank={TT_RANK})...")
    vocab_size = 32000  # Match TinyLlama exactly
    d_model = 128

    config = {"tt_rank": TT_RANK, "use_tt_embeddings": True}
    student = IndraQuantum(vocab_size, d_model, config=config).to(device)
    logging.info(f"Student Parameters: {student.get_num_params():,}")

    # 3. Load Teacher
    logging.info("Loading Teacher...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher = teacher.to(device)
    teacher.eval()

    # Bridge for Hidden State Distillation (New Feature)
    # Maps Student (128) -> Teacher (2048)
    student_dim = d_model
    teacher_dim = teacher.config.hidden_size
    # We use a simple linear projection to align the spaces
    hidden_bridge = nn.Linear(student_dim, teacher_dim).to(device)

    # 4. Optimizer & Scheduler
    # Include bridge parameters in optimizer
    optimizer = optim.AdamW(
        list(student.parameters()) + list(hidden_bridge.parameters()), lr=LR
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Resume from Checkpoint
    latest_ckpt, start_epoch = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        logging.info(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt)
        student.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        hidden_bridge.load_state_dict(checkpoint["bridge_state_dict"])
        start_epoch = checkpoint["epoch"]
        logging.info(f"Resumed at Epoch {start_epoch + 1}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")

    # 6. Training Loop
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    mse_loss_fn = nn.MSELoss()

    # Setup CSV Logging
    metrics_file = "logs/training_metrics_rank64.csv"
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Epoch", "Batch", "Total_Loss", "CE_Loss", "KD_Logits", "KD_Hidden"]
            )

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()

            # Forward Student (We need to extract hidden states ideally, but for now we rely on the final layer output before proj)
            # To do this properly without changing model code too much, let's assume 'student_logits' is result of projection.
            # We can't easily get pre-proj state without hooking.
            # Let's use the embedding output? No, that's too early.
            # Let's just use Logits KD for now to keep it simple, OR modify model to return hidden.
            # Given user constraints, let's stick to Logits KD + CE but with RANK 64 which is the main fix.
            # Adding Hidden KD requires model refactor. Let's do that if Rank 64 fails.

            # Student Forward
            student_logits = student(input_ids)

            with torch.no_grad():
                teacher_out = teacher(input_ids)
                teacher_logits = teacher_out.logits

            # 1. CE Loss (Hard Labels)
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_ce = nn.CrossEntropyLoss()(
                shift_logits.view(-1, vocab_size), shift_labels.view(-1)
            )

            # 2. KD Loss (Soft Labels)
            T = 2.0
            loss_kd = kl_loss_fn(
                torch.nn.functional.log_softmax(student_logits / T, dim=-1),
                torch.nn.functional.softmax(teacher_logits / T, dim=-1),
            ) * (T * T)

            # Combined Loss
            alpha = 0.5
            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            loss.backward()
            optimizer.step()

            # Metrics
            loss_valid = 0  # Placeholder for hidden loss if we added it

            # Log to CSV
            if batch_idx % 10 == 0:
                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch + 1,
                            batch_idx,
                            loss.item(),
                            loss_ce.item(),
                            loss_kd.item(),
                            0.0,
                        ]
                    )

            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.2f}",
                    "ce": f"{loss_ce.item():.2f}",
                    "kd": f"{loss_kd.item():.2f}",
                }
            )

        avg_loss = total_loss / len(loader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        scheduler.step()

        # Save Checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            ckpt_path,
        )
        logging.info(f"Saved checkpoint to {ckpt_path}")

    logging.info("Training Complete.")


if __name__ == "__main__":
    train()
