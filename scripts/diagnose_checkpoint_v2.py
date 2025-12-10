import torch
import sys
import os
import math

# Add root
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from indra.models.output import HybridOutput


def diagnose():
    print("--- DIAGNOSING CHECKPOINT ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config matching the training script
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
    ckpt_dir = "checkpoints/tiny_english"
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not checkpoints:
        print("No checkpoints found!")
        return

    checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    latest = checkpoints[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Loading {ckpt_path}...")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 1. Check Embedding Magnitudes
    print("\n[1. Embedding Stats]")
    embeddings = model.token_embedding
    # M = softplus(raw_mag) + 1e-4
    raw_mag_weight = embeddings.raw_mag.weight.detach()
    M = torch.nn.functional.softplus(raw_mag_weight) + 1e-4

    avg_mag = M.mean().item()
    max_mag = M.max().item()
    min_mag = M.min().item()
    print(f"Avg Embedding Magnitude: {avg_mag:.4f}")
    print(f"Max Embedding Magnitude: {max_mag:.4f}")
    print(f"Min Embedding Magnitude: {min_mag:.4f}")

    # 2. Check Logit Scale
    print("\n[2. Logit Stats (Dummy Input)]")
    input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)
    with torch.no_grad():
        logits, final_mag, final_phase = model(input_ids)

    # shape (1, 10, 32000)
    avg_logit = logits.mean().item()
    max_logit = logits.max().item()
    std_logit = logits.std().item()

    print(f"Avg Logit Value: {avg_logit:.4f}")
    print(f"Max Logit Value: {max_logit:.4f}")
    print(f"Std Logit Value: {std_logit:.4f}")

    print(f"Expected Scale for Softmax: ~1.0 to 10.0")
    if std_logit < 0.5:
        print(
            ">> WARNING: Logits are too compressed. Softmax will be uniform (High Entropy)."
        )

    # 3. Check Alpha
    print("\n[3. Hybrid Output Alpha]")
    alpha = model.output_layer.alpha.item()
    alpha_val = torch.nn.functional.softplus(model.output_layer.alpha).item()
    print(f"Raw Alpha: {alpha:.4f}")
    print(f"Effective Alpha: {alpha_val:.4f}")


if __name__ == "__main__":
    diagnose()
