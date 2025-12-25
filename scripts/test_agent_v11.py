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

        # Generation Loop with Sampling & Repetition Penalty
        print("\nIndra: ", end="", flush=True)

        # Hyperparameters
        temperature = 0.7
        top_k = 50
        top_p = 0.9
        repetition_penalty = 1.2
        max_new_tokens = 100

        with torch.no_grad():
            generated_ids = input_ids.clone()

            for i in range(max_new_tokens):
                # Crop context if growing too large
                context = generated_ids[:, -config["model"]["max_seq_len"] :]

                outputs, _, _ = model(context)
                next_token_logits = outputs[:, -1, :]

                # 1. Repetition Penalty
                # penalize tokens that have already been generated
                for token_id in set(generated_ids[0].tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

                # 2. Temperature
                next_token_logits = next_token_logits / temperature

                # 3. Top-K Filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k)
                    min_v = v[:, -1]
                    next_token_logits[next_token_logits < min_v.unsqueeze(1)] = -float(
                        "Inf"
                    )

                # 4. Top-P (Nucleus) Filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(
                        dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                # 5. Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Print token
                word = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(f"{word}", end="", flush=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        print("\n")


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
