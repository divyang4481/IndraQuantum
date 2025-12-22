import torch
from transformers import AutoTokenizer
import sys
import os

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indra.models.indra_v11 import IndraV11
import yaml


def test_model(checkpoint_path, config_path):
    device = torch.device("cpu")  # FORCE CPU to avoid crashing the training run
    print(f"Testing Checkpoint: {checkpoint_path} on {device}")

    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load Tokenizer (Llama)
    tokenizer = AutoTokenizer.from_pretrained(config["distillation"]["teacher_model"])

    # Init Model
    model = IndraV11(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        max_seq_len=config["model"]["max_seq_len"],
        tie_word_embeddings=config["model"]["tie_word_embeddings"],
    ).to(device)

    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model Loaded. Step: {checkpoint.get('step', 'Unknown')}")

    # Interactive Loop
    print("\n--- INDRA V11 TEST CONSOLE ---")
    print("Type 'quit' to exit.")

    while True:
        prompt = input("\nUser: ")
        if prompt.lower() in ["quit", "exit"]:
            break

        # Format prompt (OpenHermes Style)
        # <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n

        # Manually constructing template to be safe
        # Llama-3 style might differ slightly but let's stick to standard chat format the tokenizer recognizes
        # Or simplistic:
        # Use the Tokenizer's Built-in Chat Template (Matches Training)
        messages = [{"role": "user", "content": prompt}]

        try:
            # Llama-3 and Modern Tokenizers supports this
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            # Fallback for models without chat template (Rare now)
            print(f"Warning: Chat Template failed ({e}). using raw concatenation.")
            input_text = f"User: {prompt}\nAssistant:"

        # Llama-3 specific handling if no template found (Manual)
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is None:
            input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            generated_ids = input_ids.clone()

            print("\nIndra: ", end="", flush=True)
            for i in range(50):  # Force 50 tokens
                outputs, _, _ = model(generated_ids)
                next_token_logits = outputs[:, -1, :]

                # Apply temperature to break "safe" loops
                next_token_logits = next_token_logits / 0.8

                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                # Print token
                word = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(f"{word}", end="", flush=True)  # Print clean word

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        print("\n\n--- DEBUG INFO ---")
        print(f"Propagated Text: {repr(input_text)}")
        print(f"Input IDs: {input_ids.tolist()}")
        print("------------------\n")


if __name__ == "__main__":
    # Auto-detect latest checkpoint if not provided
    run_dir = "runs/IndraV11-GPUDistill-Llama_20251219-130308"

    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(run_dir) if f.startswith("checkpoint")]
    if not checkpoints:
        print("No checkpoints found.")
        sys.exit()

    latest_ckpt = sorted(
        checkpoints, key=lambda x: int(x.split("step")[1].split(".")[0])
    )[-1]
    ckpt_path = os.path.join(run_dir, latest_ckpt)

    config_path = "training/config_agent_v11_gpu.yaml"

    test_model(ckpt_path, config_path)
