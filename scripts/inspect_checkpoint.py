import torch
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.indra_v5 import IndraV5


def inspect_ckpt(ckpt_path):
    print(f"Inspecting: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("File not found.")
        return

    device = "cpu"
    state_dict = torch.load(ckpt_path, map_location=device)

    # 1. Check Embedding Norms
    # Token Reals
    tok_r = state_dict.get("embedding.embed_real.weight")
    tok_i = state_dict.get("embedding.embed_imag.weight")

    pos_r = state_dict.get("pos_embedding.embed_real.weight")
    pos_i = state_dict.get("pos_embedding.embed_imag.weight")

    if tok_r is not None:
        tok_mag = torch.sqrt(tok_r**2 + tok_i**2)
        mean_tok_mag = tok_mag.mean().item()
        print(f"Mean Token Mag: {mean_tok_mag:.4f}")

    if pos_r is not None:
        pos_mag = torch.sqrt(pos_r**2 + pos_i**2)
        mean_pos_mag = pos_mag.mean().item()
        print(f"Mean Pos Mag:   {mean_pos_mag:.4f}")

    if tok_r is not None and pos_r is not None:
        print(f"Ratio (Tok/Pos): {mean_tok_mag / mean_pos_mag:.2f}")

    # 2. Check Layer Rotation (Symplectic)
    # See if it learned anything or exploded
    for key in state_dict.keys():
        if "layer_rotation" in key:
            rot = state_dict[key]
            print(f"{key}: Mean={rot.mean().item():.4f}, Std={rot.std().item():.4f}")

    # 3. Check Head Weights
    head_r = state_dict.get("head.weight_real")
    if head_r is not None:
        print(f"Head Norm: {head_r.norm().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    inspect_ckpt(args.checkpoint)
