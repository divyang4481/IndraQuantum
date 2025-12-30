import argparse
import sys
import os
import yaml
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    teacher_name = config["distillation"]["teacher_model"]
    print(f"Loading Tokenizer: {teacher_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)

    print("Loading Dataset...")
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["subset_name"],
        split="train",
        streaming=True,  # Stream to be fast
    )

    lengths = []
    print("Analyzing first 1000 samples...")
    for i, item in enumerate(dataset):
        if i >= 1000:
            break

        # Approximate check using chat template logic
        if "conversations" in item:
            msgs = [
                {
                    "role": "user" if m["from"] == "human" else "assistant",
                    "content": m["value"],
                }
                for m in item["conversations"]
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            ids = tokenizer(text).input_ids
            lengths.append(len(ids))

    lengths = np.array(lengths)
    print(f"\n--- Data Length Analysis ---")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print(f"Median Length: {np.median(lengths):.2f}")
    print(f"Max Length: {np.max(lengths)}")
    print(f"Min Length: {np.min(lengths)}")

    current_max = config["model"]["max_seq_len"]
    truncated_count = np.sum(lengths > current_max)
    truncated_pct = 100 * truncated_count / len(lengths)

    print(f"\nCurrent max_seq_len: {current_max}")
    print(f"Samples > {current_max}: {truncated_count} ({truncated_pct:.1f}%)")

    print("\n--- Content of Short Samples (< 256) ---")
    shown = 0
    for i, item in enumerate(dataset):
        if shown >= 5:
            break
        if "conversations" in item:
            msgs = [
                {
                    "role": ("user" if m["from"] == "human" else "assistant"),
                    "content": m["value"],
                }
                for m in item["conversations"]
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            if len(tokenizer(text).input_ids) < current_max:
                print(f"\n[Sample {shown}] Len: {len(tokenizer(text).input_ids)}")
                print(f"User: {msgs[0]['content'][:100]}...")
                print(f"Assistant: {msgs[-1]['content'][:100]}...")
                shown += 1

    if truncated_pct > 20:
        print(
            "\n[CRITICAL WARNING] High truncation rate! The model is losing context (Prompts) in training."
        )
    else:
        print("\nTruncation rate seems acceptable.")


if __name__ == "__main__":
    main()
