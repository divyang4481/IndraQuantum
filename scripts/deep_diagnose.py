import torch
import sys
import os
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2

checkpoint_path = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


def deep_diagnose():
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found.")
        return

    print("--- DEEP DIAGNOSIS START ---")

    # 1. Load Model
    model = IndraQuantumPhase2(32000, 128).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 2. Inspect Initial Embeddings
    print("\n[Embedding Stats]")
    emb = model.token_embedding
    with torch.no_grad():
        raw_mag = emb.raw_mag.weight
        raw_phase = emb.raw_phase.weight

        print(
            f"Raw Mag: mean={raw_mag.mean():.4f}, std={raw_mag.std():.4f}, min={raw_mag.min():.4f}, max={raw_mag.max():.4f}"
        )
        print(f"Raw Phase: mean={raw_phase.mean():.4f}, std={raw_phase.std():.4f}")

        real_mag = F.softplus(raw_mag) + 1e-4
        print(f"Actual Magnitude: mean={real_mag.mean():.4f}, max={real_mag.max():.4f}")

    # 3. Inspect Output Layer
    print("\n[Output Layer Stats]")
    out_layer = model.output_layer
    alpha = F.softplus(out_layer.alpha).item() + 1e-4
    print(f"Alpha: {alpha:.6f}")

    # 4. Forward Pass Trace for 'The'
    print("\n[Forward Pass Trace: 'The']")
    input_str = "The "
    input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
    print(f"Input IDs: {input_ids.cpu().numpy()}")

    with torch.no_grad():
        mag, phase = emb(input_ids)
        print(
            f"Token 'The' Mag: {mag[0,0].mean().item():.4f}, Phase: {phase[0,0].mean().item():.4f}"
        )

        # Manually verify get_complex_vector
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        print(f"Real part mean: {real.mean().item():.4f}")

        # Run through model (simplified trace)
        # We can't easily hook everything without modifying code, but we can look at output logits
        logits, h_mag, h_phase = model(input_ids)
        last_logits = logits[0, -1]

        print(
            f"Logits Stats: mean={last_logits.mean():.4f}, std={last_logits.std():.4f}, min={last_logits.min():.4f}, max={last_logits.max():.4f}"
        )

        # Top Predictions
        probs = F.softmax(last_logits, dim=-1)
        top_probs, top_idxs = torch.topk(probs, 10)

        print("\nTop 10 Predictions:")
        for p, idx in zip(top_probs, top_idxs):
            tok = tokenizer.decode([idx.item()])
            print(f"  {idx.item()}: '{tok}' ({p.item():.4f})")

        # Check specific tokens from the user's bad output
        bad_token_id = 22915  # 'nederbörd' approx check if exists or similar
        print(f"\nCheck bad token ID {bad_token_id}:")
        if bad_token_id < 32000:
            print(f"  Logit for {bad_token_id}: {last_logits[bad_token_id].item():.4f}")

    print("\n--- DEEP DIAGNOSIS END ---")


if __name__ == "__main__":
    deep_diagnose()
