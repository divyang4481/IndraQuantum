import torch
import time
import os
import sys
import glob
import re
import numpy as np
import json

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
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
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

    CHECKPOINT_DIR = "checkpoints/phase2_stage2"

    # Find latest checkpoint
    checkpoints = glob.glob(
        os.path.join(CHECKPOINT_DIR, "checkpoint_stage2_epoch_*.pt")
    )
    if not checkpoints:
        print("No checkpoints found")
        return

    def get_epoch(path):
        try:
            return int(re.search(r"epoch_(\d+).pt", path).group(1))
        except:
            return -1

    latest_ckpt = max(checkpoints, key=get_epoch)

    # Initialize Model
    vocab_size = 32000
    d_model = 128
    model = IndraQuantumPhase2(vocab_size, d_model).to(device)

    # Load Weights
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Parameter Count
    total, trainable = count_parameters(model)
    model_size_mb = total * 4 / 1024 / 1024

    # 2. Latency
    avg_lat, std_lat, min_lat, max_lat = benchmark_latency(model, device)

    results = {
        "checkpoint": latest_ckpt,
        "total_params": total,
        "trainable_params": trainable,
        "model_size_mb": model_size_mb,
        "latency_ms_avg": avg_lat,
        "latency_ms_std": std_lat,
        "latency_ms_min": min_lat,
        "latency_ms_max": max_lat,
        "tokens_per_sec": 1000 / avg_lat * 128,
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
