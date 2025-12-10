import torch
import sys
import os
from transformers import AutoTokenizer

# Add root to path so we can import indra
sys.path.append(os.path.abspath(".."))
# Fix path if we are running from scripts/
if os.path.basename(os.getcwd()) == "scripts":
    sys.path.append(os.path.abspath(".."))
else:
    sys.path.append(os.path.abspath("."))

from indra.models.quantum_model_v2 import IndraQuantumPhase2

# --- CONFIG ---
checkpoint_path = "../checkpoints/phase2_v4_unified/checkpoint_latest.pt"
prompts = [
    "The future of AI is",
    "Quantum mechanics explains",
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------

if os.path.exists(checkpoint_path):
    print(f"Loading {checkpoint_path}...")
    try:
        # Initialize Model (Masking fixed architecture)
        # vocab_size 32000, d_model 128 (matches training script assumption)
        model = IndraQuantumPhase2(32000, 128).to(device)

        # Load with weights_only=False because custom classes might be pickled
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older torch versions
            ckpt = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        # Check alpha value
        if hasattr(model, "output_layer") and hasattr(model.output_layer, "alpha"):
            alpha_val = (
                torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4
            )
            print(f"Current Hybrid Alpha: {alpha_val:.6f}")

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        for p in prompts:
            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)
            print(f"\nPrompt: {p.strip()}")
            print("Output: ", end="")

            # Generate with Sampling
            with torch.no_grad():
                output_ids = input_ids.clone()
                for _ in range(50):
                    # pass Mask=None for inference
                    logits, _, _ = model(output_ids, attention_mask=None)
                    next_token_logits = logits[:, -1, :]

                    # Repetition Penalty
                    repetition_penalty = 1.2
                    for token_id in set(output_ids[0].tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty

                    # Apply Temperature
                    temperature = 0.7
                    probs = torch.nn.functional.softmax(
                        next_token_logits / temperature, dim=-1
                    )

                    # Sample
                    next_token = torch.multinomial(probs, num_samples=1)

                    output_ids = torch.cat([output_ids, next_token], dim=1)
                    token_str = tokenizer.decode(next_token[0])
                    print(token_str, end="", flush=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            print()

    except Exception as e:
        print(f"Error running model: {e}")
        import traceback

        traceback.print_exc()
else:
    print(f"Checkpoint not found: {checkpoint_path}")
