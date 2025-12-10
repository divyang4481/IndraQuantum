import torch
import torch.nn.functional as F
import sys
import os
from transformers import AutoTokenizer

sys.path.append(os.path.abspath("."))
from indra.models.quantum_model_v2 import IndraQuantumPhase2

CKPT = "checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def diagnose_components():
    print(f"--- COMPONENT DIAGNOSIS ---")

    model = IndraQuantumPhase2(32000, 128).to(DEVICE)
    try:
        if os.path.exists(CKPT):
            print(f"Loading {CKPT}...")
            try:
                ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
            except:
                ckpt = torch.load(CKPT, map_location=DEVICE)

            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        else:
            print("Checkpoint not found!")
            return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    final_r, final_i = None, None

    def hook(m, i, o):
        nonlocal final_r, final_i
        final_r, final_i = o

    if hasattr(model, "layers") and len(model.layers) > 0:
        model.layers[-1].register_forward_hook(hook)
    else:
        print("Model has no layers attribute?")
        return

    prompt = "The future of"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits, _, _ = model(ids)

        if final_r is None:
            print("Hook failed to capture.")
            return

        r = final_r[0, -1]
        i = final_i[0, -1]

        emb = model.token_embedding
        out = model.output_layer

        raw_mag = emb.raw_mag.weight
        raw_phase = emb.raw_phase.weight

        M = F.softplus(raw_mag) + 1e-4
        E_real = M * torch.cos(raw_phase)

        h_real = out.proj_real(r)
        h_mag = torch.sqrt(r**2 + i**2 + 1e-6)

        l_real = torch.matmul(h_real, E_real.t())
        l_mag = torch.matmul(h_mag, M.t())

        alpha = F.softplus(out.alpha) + 1e-4

        l_comb = l_real + alpha * l_mag

        print(f"\nPrompt: '{prompt}'")
        print(f"Alpha: {alpha.item():.6f}")

        print(f"\n[STATS]")
        print(
            f"Logits Real: mean={l_real.mean():.2f}, std={l_real.std():.2f}, range=[{l_real.min():.2f}, {l_real.max():.2f}]"
        )
        print(
            f"Logits Mag : mean={l_mag.mean():.2f}, std={l_mag.std():.2f}, range=[{l_mag.min():.2f}, {l_mag.max():.2f}]"
        )
        print(
            f"Alpha*Mag  : mean={(alpha*l_mag).mean():.2f}, std={(alpha*l_mag).std():.2f}"
        )

        def show(name, l):
            probs = F.softmax(l, dim=-1)
            v, idx = torch.topk(probs, 5)
            print(f"\n{name} Top 5:")
            for p, idx in zip(v, idx):
                try:
                    tok = tokenizer.decode([idx.item()])
                except:
                    tok = "<err>"
                print(f"  {idx.item()}: '{tok}' ({p.item():.4f})")

        show("REAL Channel", l_real)
        show("MAG Channel", l_mag)
        show("Hybrid Output", l_comb)


if __name__ == "__main__":
    diagnose_components()
