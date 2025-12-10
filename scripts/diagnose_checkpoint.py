import torch
import sys
import os
from transformers import AutoTokenizer
import torch.nn.functional as F

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indra.models.quantum_model_v2 import IndraQuantumPhase2


def diagnose():
    checkpoint_path = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Loading {checkpoint_path}...")
    try:
        model = IndraQuantumPhase2(32000, 128).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        # print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"PAD TOKEN ID: {tokenizer.pad_token_id}")
    print(f"EOS TOKEN ID: {tokenizer.eos_token_id}")

    # 4. Forward Pass Diagnostic
    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get internals
        mag, phase = model.token_embedding(input_ids)

        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = model.position_embedding(pos)

        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        real = real + pos_emb

        for layer in model.layers:
            real, imag = layer(real, imag, mask=None)

        # Helper function
        _compute_geometry = getattr(
            sys.modules["indra.models.output"], "_compute_geometry", None
        )
        if _compute_geometry is None:
            # Fallback if not importable
            def _compute_geometry(raw_mag, raw_phase):
                M = F.softplus(raw_mag) + 1e-4
                E_real = M * torch.cos(raw_phase)
                return M, E_real

        M, E_real = _compute_geometry(
            model.token_embedding.raw_mag.weight, model.token_embedding.raw_phase.weight
        )

        h_real = model.output_layer.proj_real(real)
        h_mag = torch.sqrt(real**2 + imag**2 + 1e-6)

        logits_real = torch.matmul(h_real, E_real.t())
        logits_mag = torch.matmul(h_mag, M.t())

        # Override alpha for test
        alpha_val = 0.01
        print(f"\nForcing Alpha to: {alpha_val}")

        logits = logits_real + (alpha_val * logits_mag)

        last_token_logits = logits[0, -1, :]
        top_k = torch.topk(last_token_logits, 5)

        print(f"\nPrompt: {prompt}")
        print("Top 5 predictions with alpha=0.01:")
        for i in range(5):
            idx = top_k.indices[i].item()
            score = top_k.values[i].item()
            token_str = tokenizer.decode([idx])
            print(f"  {i+1}: '{token_str}' (ID: {idx}) -> Logit: {score:.4f}")

    # Generate test with modified weights manual injection not possible for module call
    # We can only simulate generation by calling model, BUT model uses stored alpha.
    # So we cannot easily test generation with new alpha unless we set it in the model.
    model.output_layer.alpha.data.fill_(0.01)

    print("\nGeneration: ", end="")
    output_ids = input_ids.clone()
    for _ in range(20):
        logits, _, _ = model(output_ids, attention_mask=None)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        print(tokenizer.decode(next_token[0]), end="", flush=True)
    print()


if __name__ == "__main__":
    with open("diagnosis_log.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            diagnose()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = sys.__stdout__
