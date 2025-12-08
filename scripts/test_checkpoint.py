import torch
import sys
import os
from transformers import AutoTokenizer

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.quantum_model_v2 import IndraQuantumPhase2


def test_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    vocab_size = 32000
    d_model = 128
    model = IndraQuantumPhase2(vocab_size, d_model).to(device)

    checkpoint_path = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please wait for Epoch 1 to complete or specify a valid checkpoint.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = [
        "Explain quantum mechanics in simple terms:",
        "The meaning of life is",
        "def fibonacci(n):",
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a poem about AI.\n\n### Response:",
    ]

    for p in prompts:
        input_ids = tokenizer.encode(p, return_tensors="pt").to(device)
        print(f"\nPrompt: {p.strip()}")

        # Generate
        with torch.no_grad():
            output_ids = input_ids.clone()
            for _ in range(100):  # Max 100 new tokens
                outputs, _, _ = model(
                    output_ids,
                    attention_mask=(output_ids != tokenizer.pad_token_id).long(),
                )
                # outputs is (logits, mag, phase)
                # logits: (B, T, V)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)

                output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated:\n{gen_text}\n{'-'*40}")


if __name__ == "__main__":
    test_generation()
