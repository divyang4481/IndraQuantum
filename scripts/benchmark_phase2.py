import torch
import time
import os
import sys
import glob
import re
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.quantum_model_v2 import IndraQuantumPhase2


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def benchmark_latency(model, device, runs=100, batch_size=1, seq_len=128):
    model.eval()
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len)).to(device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    print(f"Benchmarking ({runs} runs, batch={batch_size}, seq_len={seq_len})...")
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(dummy_input)
            end = time.time()
            latencies.append((end - start) * 1000)  # ms

    return np.mean(latencies), np.std(latencies), np.min(latencies), np.max(latencies)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    CHECKPOINT_DIR = "checkpoints/phase2_stage2"

    # Find latest checkpoint
    checkpoints = glob.glob(
        os.path.join(CHECKPOINT_DIR, "checkpoint_stage2_epoch_*.pt")
    )
    if not checkpoints:
        print("No checkpoints found in phase2_stage2 directory.")
        return

    def get_epoch(path):
        try:
            return int(re.search(r"epoch_(\d+).pt", path).group(1))
        except:
            return -1

    latest_ckpt = max(checkpoints, key=get_epoch)
    print(f"Loading checkpoint: {latest_ckpt}")

    # Initialize Model
    vocab_size = 32000
    d_model = 128
    model = IndraQuantumPhase2(vocab_size, d_model).to(device)

    # Load Weights
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Parameter Count
    total, trainable = count_parameters(model)
    print("\n--- Model Statistics ---")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size (MB): {total * 4 / 1024 / 1024:.2f} MB (assuming float32)")

    # 2. Latency
    avg_lat, std_lat, min_lat, max_lat = benchmark_latency(model, device)
    print("\n--- Latency Benchmark ---")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Std Dev: {std_lat:.2f} ms")
    print(f"Min: {min_lat:.2f} ms")
    print(f"Max: {max_lat:.2f} ms")
    print(f"Tokens/sec (approx): {1000/avg_lat * 128:.2f} tokens/s (batch=1, seq=128)")


if __name__ == "__main__":
    main()
