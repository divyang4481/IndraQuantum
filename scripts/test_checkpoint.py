import os
import glob
import re
import yaml
import torch
import argparse
import sys
from transformers import AutoTokenizer

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v5 import IndraV5


def load_latest_checkpoint(run_dir):
    # Find all checkpoint_step*.pt files
    pattern = os.path.join(run_dir, "checkpoint_step*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        # Try finding resume point
        pattern2 = os.path.join(run_dir, "*.pt")
        checkpoints = glob.glob(pattern2)
        if not checkpoints:
            print(f"No checkpoints found in {run_dir}")
            return None, 0

    # Filter out _ema.pt unless we want them? Let's use EMA if available!
    # Actually, keep it simple: prefer standard first.
    std_checkpoints = [c for c in checkpoints if "_ema" not in c]
    if not std_checkpoints:
        std_checkpoints = checkpoints  # Fallback to EMA only

    # Sort by step
    def get_step(path):
        base = os.path.basename(path)  # checkpoint_step123.pt
        num = re.findall(r"\d+", base)
        return int(num[0]) if num else 0

    latest = max(std_checkpoints, key=get_step)
    print(f"Latest checkpoint: {latest}")
    return latest, get_step(latest)


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    device="cuda",
):
    model.eval()

    # Simple Chat Template
    messages = [{"role": "user", "content": prompt}]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    print(f"\nIndra ({device}): ", end="", flush=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            # outputs is (logits, mag, phase)
            next_token_logits = outputs[0][:, -1, :]  # [B, V]

            # Debug Stats (First token only)
            if _ == 0:
                vals, ids = torch.topk(next_token_logits, 5)
                print(f"\n[DEBUG] Top 5 Logits: {vals[0].tolist()}")
                print(f"[DEBUG] Top 5 IDs: {ids[0].tolist()}")
                decoded = [tokenizer.decode([i]) for i in ids[0].tolist()]
                print(f"[DEBUG] Top 5 Tokens: {decoded}")

                # Check for NaNs
                if torch.isnan(next_token_logits).any():
                    print("[CRITICAL] NaN Logits Detected!")

            # Temperature
            next_token_logits = next_token_logits / temperature

            # Top-K
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Print
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Context Limit check
            if (
                input_ids.shape[1] > model.pos_embedding.embed_real.weight.shape[0]
            ):  # Max seq len
                break
    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, help="Directory containing checkpoints")
    parser.add_argument("--config", type=str, default="training/config_agent_v1.yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    # Find Run Dir if not provided
    if not args.run_dir:
        # Look in runs/ and pick latest
        runs = glob.glob("runs/*")
        if not runs:
            print("No runs found.")
            return
        args.run_dir = max(runs, key=os.path.getmtime)
        print(f"Auto-selected latest run: {args.run_dir}")

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load Tokenizer
    teacher_name = config["distillation"]["teacher_model"]
    print(f"Loading Tokenizer ({teacher_name})...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)

    # Load Model
    # CRITICAL: Verify Vocab Size matches Config
    config_vocab = config["model"].get("vocab_size", 151936)
    real_vocab = tokenizer.vocab_size
    print(f"Tokenizer Vocab: {real_vocab} | Config Vocab: {config_vocab}")
    if real_vocab != config_vocab:
        print(
            f"WARNING: Vocab mismatch! Using Config: {config_vocab}. Note: Qwen vocab is often 151936."
        )

    vocab_size = config_vocab

    print("Initializing IndraV5...")
    model = IndraV5(
        vocab_size=config["model"].get("vocab_size", 151936),  # Default Qwen
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=0.0,
        max_seq_len=config["model"]["max_seq_len"],
        tie_word_embeddings=config["model"].get("tie_word_embeddings", False),
    ).to(device)

    # Load Checkpoint
    ckpt_path, step = load_latest_checkpoint(args.run_dir)
    if ckpt_path:
        print(f"Loading Checkpoint: {ckpt_path} (Step {step})")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Warning: No checkpoint loaded. Using random weights.")

    # Interactive Loop
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ["exit", "quit"]:
                break

            generate(model, tokenizer, prompt, device=device)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
