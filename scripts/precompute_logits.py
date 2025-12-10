import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import sys
from tqdm import tqdm
import argparse
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.setup_teacher import setup_teacher_model


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


def precompute_logits(batch_size=32, top_k=100, output_dir="data/precomputed_logits"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data
    data_path = "data/mixed_train.pkl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    print("Loading data...")
    with open(data_path, "rb") as f:
        train_data = pickle.load(f)

    n_samples = len(train_data)
    # Assuming fixed seq len 1024 for now
    seq_len = 1024

    dataset = SimpleDataset(train_data)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )
    print(f"Data loaded: {n_samples} samples.")

    # 2. Prepare Output Files (Memmap)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    values_path = os.path.join(output_dir, "logits_values.npy")
    indices_path = os.path.join(output_dir, "logits_indices.npy")

    # Initialize memmaps
    print(f"Allocating memmaps at {output_dir}...")
    fp_values = np.memmap(
        values_path, dtype="float16", mode="w+", shape=(n_samples, seq_len, top_k)
    )
    fp_indices = np.memmap(
        indices_path, dtype="int16", mode="w+", shape=(n_samples, seq_len, top_k)
    )

    # 3. Load Teacher
    print("Loading Teacher Model...")
    teacher, _ = setup_teacher_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    teacher.to(device)

    # 4. Processing Loop
    print("Starting inference...")
    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Pre-computing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            current_batch_size = input_ids.size(0)

            # Forward pass
            outputs = teacher(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]

            # Take Top-K
            topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)  # [B, T, K]

            # Transfer to CPU numpy
            vals_np = topk_values.half().cpu().numpy()
            inds_np = topk_indices.to(torch.int16).cpu().numpy()

            # Write to memmap
            # Handle variable sequence lengths if any?
            # If train_data is all padded to 512, we are good.
            # If input_ids is [B, T], T might vary?
            # mixed_train.pkl usually has fixed length chunks or padded.
            # Let's check dimensions carefully.

            b_seq_len = vals_np.shape[1]
            if b_seq_len != seq_len:
                # If smaller, pad? If larger, truncate?
                # Ideally dataset should be uniform.
                # If varying, memmap structure [N, 512, K] relies on padding.
                if b_seq_len < seq_len:
                    # Pad with 0 for values, 0 for indices (or -1?)
                    pad_v = np.zeros(
                        (current_batch_size, seq_len - b_seq_len, top_k),
                        dtype="float16",
                    )
                    pad_i = np.zeros(
                        (current_batch_size, seq_len - b_seq_len, top_k), dtype="int16"
                    )
                    vals_np = np.concatenate([vals_np, pad_v], axis=1)
                    inds_np = np.concatenate([inds_np, pad_i], axis=1)
                else:
                    vals_np = vals_np[:, :seq_len, :]
                    inds_np = inds_np[:, :seq_len, :]

            fp_values[start_idx : start_idx + current_batch_size] = vals_np
            fp_indices[start_idx : start_idx + current_batch_size] = inds_np

            start_idx += current_batch_size

            # Flush periodically
            if start_idx % (batch_size * 10) == 0:
                fp_values.flush()
                fp_indices.flush()

    fp_values.flush()
    fp_indices.flush()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/precomputed_logits")
    args = parser.parse_args()

    precompute_logits(
        batch_size=args.batch_size, top_k=args.top_k, output_dir=args.output_dir
    )
