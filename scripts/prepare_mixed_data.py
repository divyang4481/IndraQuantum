import json
import pickle
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import os


def prepare_mixed_data():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load WikiText (Existing)
    wikitext_path = "data/wikitext/train.pkl"
    print(f"Loading WikiText from {wikitext_path}...")
    with open(wikitext_path, "rb") as f:
        wikitext_data = pickle.load(f)
    print(f"WikiText samples: {len(wikitext_data)}")

    # 2. Process Alpaca
    alpaca_path = "data/alpaca_data.json"
    print(f"Loading Alpaca from {alpaca_path}...")
    with open(alpaca_path, "r") as f:
        alpaca_raw = json.load(f)

    print(f"Alpaca raw samples: {len(alpaca_raw)}")

    alpaca_data = []
    max_length = 1024  # Matching Phase 2 seq len

    print("Tokenizing Alpaca...")
    for item in tqdm(alpaca_raw):
        # Format Instruction
        if item.get("input", ""):
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        else:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"

        # Tokenize
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        alpaca_data.append(
            {
                "input_ids": enc["input_ids"][0].tolist(),
                "attention_mask": enc["attention_mask"][0].tolist(),
            }
        )

    # 3. Combine and Shuffle
    if not isinstance(wikitext_data, list):
        print(f"WikiText is {type(wikitext_data)}, converting to list...")
        wikitext_data = list(wikitext_data)

    mixed_data = wikitext_data + alpaca_data
    random.shuffle(mixed_data)

    print(f"Total Mixed Samples: {len(mixed_data)}")

    output_path = "data/mixed_train.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(mixed_data, f)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    prepare_mixed_data()
