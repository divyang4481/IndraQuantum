import torch
import torch.nn.functional as F
import argparse
import yaml
import os
import sys
import glob
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.indra_v5 import IndraV5


def get_latest_checkpoint(runs_dir="runs"):
    """Finds the latest checkpoint in the most recent run directory."""
    if not os.path.exists(runs_dir):
        return None

    # Get all run directories
    run_paths = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    if not run_paths:
        return None

    # Sort by modification time (most recent first)
    run_paths.sort(key=os.path.getmtime, reverse=True)
    latest_run = run_paths[0]

    # Looking for .pt files
    ckpts = glob.glob(os.path.join(latest_run, "*.pt"))
    if not ckpts:
        return None

    # Sort checkponts: prefer "final_model.pt", else sort by step number
    # Simple sort by time
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def get_device(device_config):
    if device_config.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_config.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2,
    device="cpu",
):
    model.eval()

    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate
    generated = input_ids

    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward
            logits, _, _ = model(generated)
            next_token_logits = logits[:, -1, :]

            # Repetition Penalty
            # For each token in generated sequence, divide its logit by penalty (if logit > 0) or multiply (if < 0)
            # Simplified approach: Penalize logits of already generated tokens
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # Temperature
            next_token_logits = next_token_logits / temperature

            # Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n\n--- GENERATED TEXT ---\n")
    print(output_text)
    print("\n----------------------\n")
    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If None, picks latest.",
    )
    parser.add_argument(
        "--config", type=str, default="training/config_nano.yaml", help="Path to config"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Override prompt from CLI"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Penalty for repeating tokens (default: 1.2)",
    )
    args = parser.parse_args()

    # Load Config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found.")
        return

    # 1. Determine Device
    dev_conf = (
        args.device
        if args.device
        else config.get("generation", {}).get("device", "auto")
    )
    device = get_device(dev_conf)
    print(f"Using device: {device}")

    # 2. Determine Checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print("No checkpoint specified. Searching for latest...")
        checkpoint_path = get_latest_checkpoint()

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading Checkpoint: {checkpoint_path}")

    # 3. Load Model
    vocab_size = config["model"].get("vocab_size", 50257)
    model = IndraV5(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Determine Prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    else:
        prompts = config.get("generation", {}).get("prompts", ["Once upon a time"])

    # Generation settings
    gen_cfg = config.get("generation", {})
    max_tokens = gen_cfg.get("max_new_tokens", 100)
    temp = gen_cfg.get("temperature", 0.7)
    top_k = gen_cfg.get("top_k", 50)
    rep_pen = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else gen_cfg.get("repetition_penalty", 1.2)
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Run
    for p in prompts:
        generate(
            model,
            tokenizer,
            p,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            repetition_penalty=rep_pen,
            device=device,
        )


if __name__ == "__main__":
    main()
