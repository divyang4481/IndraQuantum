import torch
import sys
import os
import torch.nn.functional as F

# Add root
sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2
from transformers import AutoTokenizer


def generate_interactive():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    d_model = 256
    vocab_size = 32000
    n_layers = 4
    n_heads = 4

    print("Loading Model structure...")
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = [
        "The future of AI is",
        "The cat sat on the",
        "Once upon a time",
        "What is 2+2?",
    ]

    print("\n--- GENERATION TESTS (Sampling) ---")
    for p in prompts:
        # greedy
        out_greedy = generate(model, tokenizer, p, device, do_sample=False)
        # sample
        out_sample = generate(
            model, tokenizer, p, device, do_sample=True, temp=0.8, top_k=50
        )

        print(f"\nPrompt: {p}")
        print(f"  Greedy: {out_greedy}")
        print(f"  Sample: {out_sample}")


def generate(model, tokenizer, prompt, device, do_sample=False, temp=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = input_ids.clone()
        for _ in range(50):
            logits, _, _ = model(output)
            next_logits = logits[:, -1, :]

            if do_sample:
                next_logits = next_logits / temp
                probs = F.softmax(next_logits, dim=-1)

                # Top-k
                if top_k > 0:
                    indices_to_remove = (
                        next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    )
                    next_logits[indices_to_remove] = -float("Inf")
                    probs = F.softmax(next_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1).unsqueeze(0)

            output = torch.cat([output, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    generate_interactive()
