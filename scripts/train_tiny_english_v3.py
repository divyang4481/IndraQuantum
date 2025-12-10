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
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v3 import IndraQuantumPhase3
from indra.loss_v2 import SemanticDistillationLoss

# --- CONFIGURATION (TINY ENGLISH V3) ---
CONFIG = {
    "project_name": "IndraTiny_256_v3",
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "seq_len": 256,
    "vocab_size": 32000,
    "batch_size": 16,
    "accum_steps": 2,
    "epochs": 10,  # Start with 10 Clean Epochs
    "lr": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints/tiny_english_v3",
    "teacher_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Ensure directories
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Setup Logging
logging.basicConfig(
    filename=f"logs/{CONFIG['project_name']}.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


def train_v3():
    logging.info(f"--- STARTING {CONFIG['project_name']} (Math Theory V3) ---")
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
    from datasets import load_dataset

    logging.info("Loading FULL WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

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
    logging.info("Initializing IndraQuantum (V3 Refined)...")
    model = IndraQuantumPhase3(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
    ).to(CONFIG["device"])

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * CONFIG["epochs"]
    )
    # Using existing distillation loss (it just needs logits, mag, phase)
    criterion = SemanticDistillationLoss(temperature=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(CONFIG["device"] == "cuda"))

    # CSV Logger
    metrics_file = f"logs/{CONFIG['project_name']}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Step", "Loss", "CE", "KD", "Phase", "Mag"])

    # 5. Training Loop
    model.train()

    for epoch in range(CONFIG["epochs"]):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(CONFIG["device"])

            if input_ids.dim() == 1:
                continue

            inp = input_ids[:, :-1]
            tgt = input_ids[:, 1:].clone()
            tgt[tgt == tokenizer.pad_token_id] = -100

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

            scaler.scale(loss).backward()

            if (batch_idx + 1) % CONFIG["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix({"L": f"{loss.item() * CONFIG['accum_steps']:.3f}"})

            if batch_idx % 20 == 0:
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

        # Checkpoint
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"ckpt_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Generation
        logging.info(f"--- Epoch {epoch+1} Generation ---")
        gen = generate_sample(model, tokenizer, "The future of AI is", CONFIG["device"])
        logging.info(f"Gen: {gen}")


def generate_sample(model, tokenizer, prompt, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = input_ids.clone()
        for _ in range(50):
            logits, _, _ = model(output)
            next_logits = logits[:, -1, :]

            # Penalize
            for i in range(output.shape[1]):
                next_logits[0, output[0, i]] /= 1.2

            next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model.train()
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    train_v3()
