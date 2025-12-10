import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import math
from tqdm import tqdm
import time
import csv

# Add root to path
sys.path.append(os.path.abspath(".."))
if os.path.basename(os.getcwd()) == "scripts":
    sys.path.append(os.path.abspath(".."))
else:
    sys.path.append(os.path.abspath("."))

from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.loss_v2 import SemanticDistillationLoss

# --- CONFIGURATION (TINY ENGLISH) ---
CONFIG = {
    "project_name": "IndraTiny_256_v2",
    "d_model": 256,  # Increased for better capacity
    "n_layers": 4,  # Depth for reasoning
    "n_heads": 4,
    "seq_len": 256,  # Fast context
    "vocab_size": 32000,
    "batch_size": 16,  # Fit in 6GB
    "accum_steps": 2,  # Effective BS = 32
    "epochs": 25,  # Extended for full dataset convergence
    "lr": 5e-4,  # Standard Transformer LR
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "../checkpoints/tiny_english",
    "teacher_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Handle Paths dynamically
if os.path.basename(os.getcwd()) == "scripts":
    base_path = ".."
else:
    base_path = "."

CONFIG["checkpoint_dir"] = os.path.join(base_path, "checkpoints", "tiny_english_v2")
LOG_DIR = os.path.join(base_path, "logs")
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"{CONFIG['project_name']}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


def train_tiny():
    logging.info(f"--- STARTING {CONFIG['project_name']} ---")
    logging.info(f"Config: {CONFIG}")

    # 1. Tokenizer & Teacher
    logging.info("Loading Teacher...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_model"],
        torch_dtype=torch.float16 if CONFIG["device"] == "cuda" else torch.float32,
    ).to(CONFIG["device"])
    teacher.eval()

    # 2. Dataset
    # loading WikiText-2 or similar using huggingface datasets
    from datasets import load_dataset

    logging.info("Loading WikiText-2...")
    ds = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train"
    )  # Full dataset for better quality

    # Tokenize
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=CONFIG["seq_len"],
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": outputs["input_ids"].squeeze(),
            "attention_mask": outputs["attention_mask"].squeeze(),
        }

    tokenized_ds = ds.map(tokenize, remove_columns=["text"], batched=True)
    tokenized_ds.set_format("torch")
    loader = DataLoader(tokenized_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    logging.info(f"Dataset Size: {len(loader)} batches")

    # 3. Student Model
    logging.info("Initializing IndraQuantum (Tiny)...")
    model = IndraQuantumPhase2(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        config={"mode": "fullq"},  # Start with full quantum
    ).to(CONFIG["device"])

    # Resumption Logic
    start_epoch = 0
    checkpoints = [
        f
        for f in os.listdir(CONFIG["checkpoint_dir"])
        if f.endswith(".pt") and f.startswith("ckpt_")
    ]
    if checkpoints:
        # Sort by epoch number assuming format "ckpt_epoch_X.pt"
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        latest_ckpt = checkpoints[-1]
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], latest_ckpt)

        logging.info(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=CONFIG["device"]), strict=False
        )
        start_epoch = int(latest_ckpt.split("_")[2].split(".")[0])
        logging.info(f"Resuming at Epoch {start_epoch + 1}")
    else:
        logging.info("No checkpoints found. Starting from scratch.")

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    # Scheduler Reset: Calculate remaining steps
    remaining_epochs = CONFIG["epochs"] - start_epoch
    if remaining_epochs <= 0:
        remaining_epochs = 1

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * remaining_epochs
    )
    criterion = SemanticDistillationLoss(temperature=2.0)
    # scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["device"] == "cuda"))
    # Fix FutureWarning by using new API
    scaler = torch.amp.GradScaler("cuda", enabled=(CONFIG["device"] == "cuda"))

    # CSV Logger
    metrics_file = f"logs/{CONFIG['project_name']}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Step", "Loss", "CE", "KD", "Phase", "Mag"])

    # 5. Training Loop
    model.train()

    for epoch in range(start_epoch, CONFIG["epochs"]):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(CONFIG["device"])

            # Skip short batches if any (WikiText sometimes has small shards)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            # Shift targets
            inp = input_ids[:, :-1]
            tgt = input_ids[:, 1:].clone()
            tgt[tgt == tokenizer.pad_token_id] = -100

            # Forward
            with torch.amp.autocast("cuda", enabled=(CONFIG["device"] == "cuda")):
                # Student
                logits, mag, phase = model(
                    inp, attention_mask=(inp != tokenizer.pad_token_id).int()
                )

                # Teacher
                with torch.no_grad():
                    t_out = teacher(inp)

                # Loss
                loss, terms = criterion(
                    student_logits=logits,
                    student_mag=mag,
                    student_phase=phase,
                    teacher_logits=t_out.logits,
                    targets=tgt,
                    current_epoch=epoch,
                    total_epochs=CONFIG["epochs"],
                )
                loss = loss / CONFIG["accum_steps"]

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % CONFIG["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * CONFIG["accum_steps"]

            # Logs
            progress.set_postfix(
                {
                    "L": f"{loss.item() * CONFIG['accum_steps']:.3f}",
                    "CE": f"{terms['ce']:.3f}",
                    "KD": f"{terms['kd']:.3f}",
                }
            )

            if batch_idx % 10 == 0:
                with open(metrics_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch + 1,
                            batch_idx,
                            loss.item() * CONFIG["accum_steps"],
                            terms["ce"],
                            terms["kd"],
                            terms["phase"],
                            terms["mag"],
                        ]
                    )

        # Test Generation
        test_prompts = ["The future of AI is", "What is 2+2?"]
        logging.info(f"--- Epoch {epoch+1} Generation ---")
        for p in test_prompts:
            gen = generate_sample(model, tokenizer, p, CONFIG["device"])
            logging.info(f"Prompt: {p} -> {gen}")

        # Save Checkpoint
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"ckpt_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)


def generate_sample(model, tokenizer, prompt, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = input_ids.clone()
        for _ in range(30):
            logits, _, _ = model(output)
            next_logits = logits[:, -1, :]

            # Simple Repetition Penalty
            for i in range(output.shape[1]):
                token_id = output[0, i].item()
                next_logits[0, token_id] /= 1.2  # Penalize existing tokens

            next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model.train()
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    train_tiny()
