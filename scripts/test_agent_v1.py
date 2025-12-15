import torch
import os
import sys
import glob
import argparse
from transformers import AutoTokenizer

# Add root context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.indra_v5 import IndraV5


def load_latest_checkpoint(run_dir=None):
    if run_dir is None:
        # Find latest run in runs/
        runs = glob.glob("runs/*")
        if not runs:
            raise FileNotFoundError("No runs found in runs/")
        run_dir = max(runs, key=os.path.getmtime)
        print(f"🔹 Detected latest run: {run_dir}")

    # Find latest checkpoint in run_dir
    checkpoints = glob.glob(os.path.join(run_dir, "*.pt"))
    # Filter out "final" if you want the stepped checkpoints, or prefer final?
    # Usually "checkpoint_stepX.pt" is safest.
    step_checkpoints = [c for c in checkpoints if "checkpoint_step" in c]

    if not step_checkpoints:
        # Fallback to final
        if os.path.join(run_dir, "final_agent_v1.pt") in checkpoints:
            return os.path.join(run_dir, "final_agent_v1.pt")
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    # Sort by step number
    # "checkpoint_step100.pt" -> 100
    latest_ckpt = max(
        step_checkpoints,
        key=lambda x: int(x.split("checkpoint_step")[-1].split(".")[0]),
    )
    return latest_ckpt


def generate(
    model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, device="cuda"
):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Simple generation loop for IndraV5 (Complex Model)
    # We can't use HF generate() easily because IndraV5 signature returns (logits, mag, phase)
    # So we write a manual sampling loop.

    curr_ids = input_ids

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(curr_ids)
            # IndraV5 returns (logits, mag, phase)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Get last token logits
            next_token_logits = logits[:, -1, :]

            # Apply Temperature
            next_token_logits = next_token_logits / temperature

            # Sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Decode just the new token to stream it
            word = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(word, end="", flush=True)

            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n" + "-" * 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, default=None, help="Specific run directory (optional)"
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    args = parser.parse_args()

    # FORCE CPU for testing to avoid GPU conflict
    device = torch.device("cpu")
    print("🔹 Using Device: CPU")

    # 1. Locate Checkpoint
    ckpt_path = load_latest_checkpoint(args.run_dir)
    print(f"🔹 Loading Checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 2. Re-create Model from Config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        print("⚠️ Config not found in checkpoint! Cannot initialize model safely.")
        return

    # Helper to get teacher for Tokenizer
    teacher_name = config["distillation"]["teacher_model"]
    print(f"🔹 Loading Tokenizer: {teacher_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Detect Correct Vocab Size from Weights
    state_dict = checkpoint["model_state_dict"]
    detected_vocab = state_dict["final_bias"].shape[0]
    print(f"🔹 Detected Training Vocab Size: {detected_vocab}")

    print("🔹 Initializing IndraV5...")
    model = IndraV5(
        vocab_size=detected_vocab,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=0.0,
        max_seq_len=config["model"]["max_seq_len"],
        tie_word_embeddings=config["model"].get("tie_word_embeddings", False),
    ).to(device)

    # 4. Load Weights
    model.load_state_dict(state_dict)
    print("✅ Model Loaded Successfully")

    # 5. Run Tests
    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Explain Quantum Physics in simple terms:",
        "Write a python function to print hello world:",
    ]

    if args.prompt:
        prompts = [args.prompt]

    for p in prompts:
        generate(model, tokenizer, p, temperature=args.temp, device=device)


if __name__ == "__main__":
    main()
