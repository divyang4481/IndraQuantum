import torch
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.indra_v5 import IndraV5
from indra.losses.holographic import HolographicLoss


def check_gradients():
    print("--- IndraV5 Gradient Flow Check ---")

    # Tiny Config
    vocab_size = 100
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    max_len = 10

    model = IndraV5(
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        dropout=0.0,
        max_seq_len=max_len,
    )

    # Dummy Input: "A B A B"
    # [0, 1, 0, 1]
    input_ids = torch.tensor([[0, 1, 0, 1, 2, 3, 4, 5, 0, 0]])  # Batch 1, Len 10
    # Targets (shifted)
    labels = torch.tensor([[1, 0, 1, 2, 3, 4, 5, 0, 0, 0]])

    # Forward
    print("Forward Pass...")
    logits, mag, phase = model(input_ids)

    # Loss (Standard CE for now, ignore Holographic for grad check)
    criterion = torch.nn.CrossEntropyLoss()
    # Complex Reconstruct for Holographic API if needed, but let's just use CE on logits
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

    print(f"Loss: {loss.item()}")

    # Backward
    print("Backward Pass...")
    loss.backward()

    # Measure Gradients
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.norm().item()
        else:
            grads[name] = 0.0

    # Analysis
    emb_grad = (
        grads.get("embedding.embed_real.weight", 0)
        + grads.get("embedding.embed_imag.weight", 0)
    ) / 2
    pos_grad = (
        grads.get("pos_embedding.embed_real.weight", 0)
        + grads.get("pos_embedding.embed_imag.weight", 0)
    ) / 2
    head_grad = (
        grads.get("head.weight_real", 0) + grads.get("head.weight_imag", 0)
    ) / 2

    print("\n--- Gradient Norms ---")
    print(f"Token Embedding: {emb_grad:.6f}")
    print(f"Pos Embedding:   {pos_grad:.6f}")
    print(f"Output Head:     {head_grad:.6f}")

    if pos_grad > 0:
        ratio = emb_grad / pos_grad
        print(f"Token/Pos Ratio: {ratio:.2f}")
        if ratio > 10.0:
            print("WARNING: Positional Signal is weak compared to Token Signal.")
        elif ratio < 0.1:
            print("WARNING: Token Signal is weak compared to Positional Signal.")
        else:
            print("SUCCESS: Balanced Signal.")
    else:
        print("ERROR: Positional Gradient is 0!")

    # Layer Gradients
    print("\n--- Layer Gradients ---")
    for i in range(num_layers):
        w_real = grads.get(f"layers.{i}.ffn.w1.weight_real", 0)
        print(f"Layer {i} FFN W1: {w_real:.6f}")


if __name__ == "__main__":
    check_gradients()
