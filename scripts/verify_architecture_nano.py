import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Add root to path
sys.path.append(os.path.abspath(".."))
if os.path.basename(os.getcwd()) == "scripts":
    sys.path.append(os.path.abspath(".."))
else:
    sys.path.append(os.path.abspath("."))

from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.loss_v2 import SemanticDistillationLoss

# --- CONFIGURATION (NANO) ---
CONFIG = {
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "seq_len": 64,  # Very short for speed
    "batch_size": 16,
    "epochs": 100,  # Fast epochs
    "lr": 1e-3,  # Slightly high for fast overfit
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vocab_size": 32000,  # TinyLlama Tokenizer
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- SYNTHETIC DATA GENERATOR ---
# We want strict logic to force the model to learn relationships, not just stats.
QA_PAIRS = [
    ("What color is the sky?", "The sky is blue."),
    ("What is 2 plus 2?", "2 plus 2 is 4."),
    ("Who are you?", "I am Indra, a quantum AI."),
    ("Is fire hot or cold?", "Fire is hot."),
    ("What comes after A?", "B comes after A."),
    ("The cat sat on the?", "The cat sat on the mat."),
    ("Day follows?", "Day follows night."),
    ("Up is the opposite of?", "Up is the opposite of down."),
]


class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, qa_pairs, size=500, seq_len=64):
        self.data = []
        self.tokenizer = tokenizer

        # Repeat the pairs to fill the dataset size
        for i in range(size):
            q, a = qa_pairs[i % len(qa_pairs)]
            text = f"### Instruction:\n{q}\n\n### Response:\n{a}{tokenizer.eos_token}"
            enc = tokenizer(
                text,
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.data.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_sample(model, tokenizer, prompt, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Greedy decoding for checking memorization
        output = input_ids.clone()
        for _ in range(20):
            logits, _, _ = model(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(output[0], skip_special_tokens=True)


def train_nano():
    print(f"--- STARTING NANO VERIFICATION ON {CONFIG['device']} ---")

    # 1. Tokenizer & Teacher (Teacher needed for Loss, even if dummy for start)
    # Ideally we use a real teacher for KD, but for nano-overfit,
    # we can try to overfit just CE first to prove architecture connects.
    # To fully test IndraQuantum, we DO need the teacher logits.
    # Let's load the TinyLlama teacher on the fly (it's small).
    print("Loading Teacher (TinyLlama)...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16 if CONFIG["device"] == "cuda" else torch.float32,
    ).to(CONFIG["device"])
    teacher.eval()

    # 2. Dataset
    print("Generating Synthetic Data...")
    dataset = SyntheticDataset(tokenizer, QA_PAIRS, size=200, seq_len=CONFIG["seq_len"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 3. Student Model
    print("Initializing Nano IndraQuantum...")
    model = IndraQuantumPhase2(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
    ).to(CONFIG["device"])

    # 4. Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = SemanticDistillationLoss(temperature=2.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["device"] == "cuda"))

    # 5. Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(CONFIG["device"])
            # Create Targets (Shifted)
            targets = input_ids.clone()
            # We don't want to predict the padding
            # Mask is 0 for pad.
            # We want targets to be -100 where it's padding?
            # Or reliance on attention mask.
            # Let's simple shift:
            # In:  A B C D
            # Out: B C D E
            # We feed Input[:, :-1] and predict Input[:, 1:]

            inp = input_ids[:, :-1]
            tgt = input_ids[:, 1:].clone()

            # Mask padding in target (pad_token_id = 0 usually or 2)
            tgt[tgt == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(CONFIG["device"] == "cuda")):
                # Student Forward
                logits, mag, phase = model(inp)  # (B, T, V)

                # Teacher Forward (No Grad)
                with torch.no_grad():
                    t_out = teacher(inp)
                t_logits = t_out.logits

                # Loss
                loss, terms = criterion(
                    student_logits=logits,
                    student_mag=mag,
                    student_phase=phase,
                    teacher_logits=t_logits,
                    targets=tgt,
                    current_epoch=epoch,
                    total_epochs=CONFIG["epochs"],
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | CE: {terms['ce']:.4f} | KD: {terms['kd']:.4f} | Ph: {terms['phase']:.4f}"
            )

            # Test Generation
            test_prompt = "### Instruction:\nWhat is 2 plus 2?\n\n### Response:\n"
            gen_text = generate_sample(model, tokenizer, test_prompt, CONFIG["device"])
            print(f"   Gen: {gen_text.replace(test_prompt, '').strip()}")

            # Check Alpha
            if hasattr(model.output_layer, "alpha"):
                alpha = torch.nn.functional.softplus(model.output_layer.alpha).item()
                print(f"   Alpha: {alpha:.6f}")

    print("--- VERIFICATION COMPLETE ---")


if __name__ == "__main__":
    train_nano()
