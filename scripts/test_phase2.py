import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
import os
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.quantum_model_v2 import IndraQuantumPhase2


def test_checkpoint(
    checkpoint_path, prompt="The meaning of quantum mechanics is", temperature=0.7
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # 1. Initialize Model (V2 Architecture)
    vocab_size = 32000
    d_model = 128
    model = IndraQuantumPhase2(vocab_size, d_model).to(device)

    # 2. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Check for state dict mismatch
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded model from epoch {checkpoint['epoch']}")
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # Try strict=False if minor mismatches, but for V1 vs V2 it will fail hard
        return

    model.eval()

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 4. Generate
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print("\n--- Generating ---")
    with torch.no_grad():
        # Simple Greedy Generation loop since 'generate' might not be in V2 class yet or different
        # Let's implement a quick generation loop here to be sure

        curr_ids = input_ids.clone()
        for _ in range(50):
            # Forward
            # V2 returns logits, mag, phase
            logits, _, _ = model(curr_ids)
            next_token_logits = logits[:, -1, :]

            # Temperature Sampling
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        output_text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {output_text}")
        print("------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    args = parser.parse_args()

    test_checkpoint(args.checkpoint_path, args.prompt, args.temperature)
