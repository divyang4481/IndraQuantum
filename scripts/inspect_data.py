import pickle
import sys
import os
import torch
from transformers import AutoTokenizer


def inspect_data():
    sys.stdout.reconfigure(encoding="utf-8")
    data_path = "data/mixed_train.pkl"
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print("\n--- Sample 0 ---")
    ids = data[0]["input_ids"]
    text = tokenizer.decode(ids)
    print(f"Decoded: {text}")
    print(f"IDs: {ids[:30]}...")

    print("\n--- Sample 100 ---")
    ids = data[100]["input_ids"]
    text = tokenizer.decode(ids)
    print(f"Decoded: {text}")

    print("\n--- Sample 1000 ---")
    ids = data[1000]["input_ids"]
    text = tokenizer.decode(ids)
    print(f"Decoded: {text}")


if __name__ == "__main__":
    inspect_data()
