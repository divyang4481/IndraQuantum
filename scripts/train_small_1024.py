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
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.loss_v2 import SemanticDistillationLoss

# --- CONFIGURATION (SMALL 1024) ---
CONFIG = {
    "project_name": "IndraSmall_1024",
    "d_model": 512,  # Scaled up
    "n_layers": 6,  # Scaled up
    "n_heads": 8,  # Scaled up
    "seq_len": 1024,  # TARGET CONTEXT
    "vocab_size": 32000,
    "batch_size": 4,  # Low BS for VRAM safety with 1024 ctx
    "accum_steps": 8,  # Effective BS = 32
    "epochs": 5,  # 5 Epochs on FULL dataset
    "lr": 3e-4,  # Slower LR for stability
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints/small_1024",
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


def train_small():
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

    # 2. Dataset (FULL)
    from datasets import load_dataset

    logging.info("Loading FULL WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

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
    logging.info("Initializing IndraQuantum (Small 1024)...")
    model = IndraQuantumPhase2(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        config={"mode": "fullq"},
    ).to(CONFIG["device"])

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * CONFIG["epochs"]
    )
    criterion = SemanticDistillationLoss(temperature=2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(CONFIG["device"] == "cuda"))

    # CSV Logger
    metrics_file = f"logs/{CONFIG['project_name']}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Step", "Loss", "CE", "KD", "Phase", "Mag"])

    # Resumption Logic
    start_epoch = 0
    checkpoints = [
        f
        for f in os.listdir(CONFIG["checkpoint_dir"])
        if f.endswith(".pt") and f.startswith("ckpt_")
    ]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        latest_ckpt = checkpoints[-1]
        ckpt_path = os.path.join(CONFIG["checkpoint_dir"], latest_ckpt)
        logging.info(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=CONFIG["device"]), strict=False
        )
        start_epoch = int(latest_ckpt.split("_")[2].split(".")[0])

    # 5. Training Loop
    model.train()

    for epoch in range(start_epoch, CONFIG["epochs"]):
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(CONFIG["device"])

            # Skip short batches
            if input_ids.dim() == 1:
                continue

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

            # Logs
            progress.set_postfix(
                {
                    "L": f"{loss.item() * CONFIG['accum_steps']:.3f}",
                    "KD": f"{terms['kd']:.3f}",
                }
            )

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

        # Save Checkpoint
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

            # Repetition Penalty
            for i in range(output.shape[1]):
                next_logits[0, output[0, i]] /= 1.2

            next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    model.train()
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    train_small()
