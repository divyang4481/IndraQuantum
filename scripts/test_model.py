import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v5 import IndraV5
from indra.losses.holographic import HolographicLoss


def test_v5_init():
    print("Testing IndraV5 Initialization...")

    # Tiny Config
    vocab = 100
    model = IndraV5(vocab_size=vocab, d_model=32, num_layers=2, num_heads=4, d_ff=64)
    print("Model initialized.")

    # Dummy Input
    input_ids = torch.randint(0, vocab, (2, 10))  # Batch 2, Seq 10

    # Forward
    logits, mag, phase = model(input_ids)
    z_states = torch.polar(mag, phase)
    print(f"Logits Shape: {logits.shape} (Expected [2, 10, 100])")
    print(f"State Shape: {z_states.shape} (Expected [2, 10, 32])")
    print(f"Is Complex State? {torch.is_complex(z_states)}")

    # Loss
    loss_fn = HolographicLoss()
    target_ids = torch.randint(0, vocab, (2, 10))
    loss, metrics = loss_fn(logits, z_states, targets=target_ids)

    print(f"Loss Computed: {loss.item()}")
    print(f"Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN!"
    print("Test Passed: Forward pass and Loss calculation successful.")


if __name__ == "__main__":
    test_v5_init()
