import torch
import sys
import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Add root to path so we can import indra
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2

# --- CONFIG ---
checkpoint_path = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
prompts = [
    "The future of AI is",
    "Quantum mechanics explains",
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
]
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------


def test_generation():
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading {checkpoint_path}...")
    try:
        # Initialize Model
        model = IndraQuantumPhase2(32000, 128).to(device)
        # Load with weights_only=False because custom classes might be pickled (though state_dict is safer)
        # Assuming typical state_dict save:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception:
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
        print(f"Error running inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_generation()
