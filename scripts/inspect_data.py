import pickle
import torch
from transformers import AutoTokenizer


def inspect_data():
    print("Loading data...")
    with open("data/wikitext/train.pkl", "rb") as f:
        data = pickle.load(f)[0:5]  # First 5 samples

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"Tokenizer PAD ID: {tokenizer.pad_token_id}")
    print(f"Tokenizer UNK ID: {tokenizer.unk_token_id}")
    print(f"Tokenizer EOS ID: {tokenizer.eos_token_id}")

    for i in range(5):
        item = data[i]
        # Verify item structure
        if not isinstance(item, dict):
            print(f"Warning: Item {i} is not a dict, it is {type(item)}")
            continue
        ids = item["input_ids"]
        mask = item["attention_mask"]
        print(f"\nSample {i} (Length {len(ids)}):")
        print(f"IDs: {ids[:20]} ... {ids[-20:]}")
        print(f"Mask: {mask[:20]} ... {mask[-20:]}")
        print(f"Decoded: {tokenizer.decode(ids[:50])}")

        # Check frequency of most common token
        unique, counts = torch.tensor(ids).unique(return_counts=True)
        most_common_idx = torch.argmax(counts)
        print(
            f"Most Common Token: {unique[most_common_idx]} (Count: {counts[most_common_idx]})"
        )


if __name__ == "__main__":
    inspect_data()
