import torch
import sys
import os
import argparse
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.quantum_core import IndraQuantum


def generate_text_strict(checkpoint_path, prompt, tt_rank=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Suppress loading noise

    # Init Model (Re-init for every call is safe but slow, fine for testing)
    vocab_size = 32000
    d_model = 128
    config = {"tt_rank": tt_rank, "use_tt_embeddings": True}
    model = IndraQuantum(vocab_size, d_model, config=config).to(device)

    # Load Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print(f"\n[{prompt}]")  # Bracket prompt to see it clearly

    # 1. Greedy Decoding (Argmax)
    generated = input_ids.clone()
    for _ in range(50):
        with torch.no_grad():
            logits = model(generated)
            # Penalize repetition manually if needed, but let's see raw greedy first
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Remove the prompt from the output for cleaner display
    new_text = decoded[len(prompt) :]
    print(f"GREEDY >> {new_text.strip()}")

    # 2. Top-K=5 (Stricter Sampling)
    generated = input_ids.clone()
    for _ in range(50):
        with torch.no_grad():
            logits = model(generated)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 5, dim=-1)
            next_token_idx = torch.multinomial(top_k_probs, 1)
            next_token = torch.gather(top_k_indices, -1, next_token_idx)
            generated = torch.cat([generated, next_token], dim=1)

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    new_text = decoded[len(prompt) :]
    print(f"TOP-K5 >> {new_text.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()

    print("-" * 50)
    print(f"Testing Checkpoint: {os.path.basename(args.checkpoint)}")
    print("-" * 50)

    prompts = ["The", "It is", "In 2024"]
    for p in prompts:
        generate_text_strict(args.checkpoint, p, args.rank)
