import torch
import sys
import os
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt

# Add root
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from transformers import AutoTokenizer


def debug_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config matching trained model
    d_model = 256
    vocab_size = 32000
    n_layers = 4
    n_heads = 4

    model = IndraQuantumPhase2(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        config={"mode": "fullq"},
    ).to(device)

    # Load latest checkpoint
    ckpt_dir = "checkpoints/tiny_english_v2"
    checkpoints = [
        f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f.startswith("ckpt_")
    ]
    if not checkpoints:
        print("No checkpoints found!")
        return

    checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    latest = checkpoints[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Loading {ckpt_path}...")

    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("\n[DEBUG LOGITS]")
    with torch.no_grad():
        logits, mag, phase = model(input_ids)
        # Check last token logits
        last_logits = logits[0, -1, :]  # (vocab_size)

        # 1. Top predictions
        probs = F.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)

        print(f"Top 10 Predictions for '{prompt}':")
        for p, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx.item()])
            print(f"  '{token}': {p.item():.4f} (Logit: {last_logits[idx].item():.4f})")

        # 2. Check bias
        bias = model.output_layer.bias
        print(
            f"\nBias Stats: Mean={bias.mean().item():.4f}, Max={bias.max().item():.4f}, Min={bias.min().item():.4f}"
        )

        # 3. Check Attention output norms (is signal vanishing?)
        # We can't easily hook without forward ref, but we can check hidden states magnitude
        # The model returns logits, which come from h_real and h_mag.
        # Check alpha
        print(
            f"Alpha: {model.output_layer.alpha.item():.4f} -> {F.softplus(model.output_layer.alpha).item():.4f}"
        )


if __name__ == "__main__":
    debug_generation()
