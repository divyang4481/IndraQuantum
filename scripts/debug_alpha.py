import torch
import sys
import os
from transformers import AutoTokenizer

# Add root to path
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2

# --- CONFIG ---
checkpoint_path = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
prompts = [
    "The future of AI is",
    "Quantum mechanics explains",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------


def test_generation():
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading {checkpoint_path}...")
    try:
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

        # --- DIAGNOSTICS ---
        if hasattr(model, "output_layer") and hasattr(model.output_layer, "alpha"):
            original_alpha = (
                torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4
            )
            print(f"Original Trained Alpha: {original_alpha:.6f}")

            # TEST 1: Force Alpha to near-zero to see 'Real' Semantic Learning
            print("\n>>> TEST: Overriding Alpha to 0.001 (Pure Semantics)")
            with torch.no_grad():
                # We need to set the raw parameter such that softplus(raw) ~= 0.001
                # softplus(x) = log(1+e^x). If x is very negative, softplus -> 0.
                model.output_layer.alpha.data.fill_(-10.0)

            new_alpha = (
                torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4
            )
            print(f"New Effective Alpha: {new_alpha:.6f}")

        # -------------------

        for p in prompts:
            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)
            print(f"\nPrompt: {p.strip()}")
            print("Output: ", end="")

            with torch.no_grad():
                output_ids = input_ids.clone()
                for _ in range(30):
                    logits, _, _ = model(output_ids, attention_mask=None)
                    next_token_logits = logits[:, -1, :]

                    # Lower temperature for this test to see best guesses
                    temperature = 0.5
                    probs = torch.nn.functional.softmax(
                        next_token_logits / temperature, dim=-1
                    )
                    next_token = torch.multinomial(probs, num_samples=1)

                    output_ids = torch.cat([output_ids, next_token], dim=1)
                    token_str = tokenizer.decode(next_token[0])
                    print(token_str, end="", flush=True)

                    if next_token.item() == tokenizer.eos_token_id:
                        break
            print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_generation()
