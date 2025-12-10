import torch
import sys
import os
import torch.nn.functional as F

# Add root
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v3 import IndraQuantumPhase3
from transformers import AutoTokenizer


def generate_v3():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    d_model = 256
    vocab_size = 32000
    n_layers = 4
    n_heads = 4

    print("Loading V3 Model structure...")
    model = IndraQuantumPhase3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    # Load latest checkpoint
    ckpt_dir = "checkpoints/tiny_english_v3"
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

    prompts = ["The future of AI is", "The cat sat on the", "What is 2+2?"]

    print("\n--- INDRA QUANTUM V3 GENERATION ---")
    for p in prompts:
        out = generate(model, tokenizer, p, device, penalty=1.2)
        print(f"\nPrompt: {p}")
        print(f"Result: {out}")


def generate(model, tokenizer, prompt, device, penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = input_ids.clone()
        for _ in range(50):
            # V3 returns (logits, mag, phase)
            logits, _, _ = model(output)
            next_logits = logits[:, -1, :]

            # Repetition Penalty
            for i in range(output.shape[1]):
                next_logits[0, output[0, i]] /= penalty

            next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)
            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    generate_v3()
