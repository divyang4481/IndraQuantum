import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle


def prepare_wikitext_data(output_dir="data/wikitext", seq_len=1024):
    print(f"Preparing Wikitext-2 dataset in {output_dir}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Tokenizer (using a standard one like GPT-2 or TinyLlama)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Load Dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
    )

    # Concatenate and chunk
    block_size = seq_len

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the small remainder
        total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    # Save processed data
    print("Saving processed datasets...")
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(lm_datasets["train"], f)
    with open(os.path.join(output_dir, "validation.pkl"), "wb") as f:
        pickle.dump(lm_datasets["validation"], f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(lm_datasets["test"], f)

    print("Data preparation complete.")
    print(f"Train samples: {len(lm_datasets['train'])}")
    print(f"Val samples: {len(lm_datasets['validation'])}")


if __name__ == "__main__":
    prepare_wikitext_data()
