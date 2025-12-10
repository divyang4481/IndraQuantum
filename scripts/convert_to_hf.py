import torch
import os
import argparse
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v5 import IndraV5


def convert_checkpoint_to_hf(checkpoint_path, config_path, output_dir):
    """
    Converts an IndraQuantum V5 checkpoint to a generic HF-style save.
    Since this is a custom architecture, we save:
    1. pytorch_model.bin
    2. config.json
    3. Custom code (optional, but requested 'various HF format')
    """
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Initialize Model
    # Since we don't have a registered HF Config object, we instantiate the model class directly
    # and load state dict.
    vocab_size = config["model"].get("vocab_size", 50257)  # Default GPT2
    model = IndraV5(
        vocab_size=vocab_size,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
    )

    # Check keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    if model_keys != ckpt_keys:
        print("Warning: Key mismatch.")
        print(f"Missing in model: {ckpt_keys - model_keys}")
        print(f"Missing in ckpt: {model_keys - ckpt_keys}")

    model.load_state_dict(state_dict, strict=False)

    # Save to Output Dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Weights
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # 2. Save Config (as JSON)
    # create a HF compatible config dict
    hf_config = {
        "architectures": ["IndraV5"],
        "model_type": "indra_quantum",
        "vocab_size": vocab_size,
        "n_embd": config["model"]["d_model"],
        "n_layer": config["model"]["num_layers"],
        "n_head": config["model"]["num_heads"],
        "n_inner": config["model"]["d_ff"],
        "activation_function": "complex_gelu",
        "resid_pdrop": config["model"]["dropout"],
        "embd_pdrop": config["model"]["dropout"],
        "attn_pdrop": config["model"]["dropout"],
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "summary_activation": None,
        "summary_proj_to_labels": True,
        "summary_first_dropout": 0.1,
        "use_cache": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "auto_map": {
            "AutoModel": "indra.models.indra_v5.IndraV5",
            "AutoConfig": "indra.models.configuration_indra.IndraConfig",  # Placeholder
        },
    }

    import json

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Successfully converted and saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="hf_export")
    args = parser.parse_args()

    convert_checkpoint_to_hf(args.checkpoint, args.config, args.output_dir)


if __name__ == "__main__":
    main()
