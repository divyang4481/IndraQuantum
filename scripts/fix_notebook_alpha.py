import json
import os

notebook_path = "notebooks/monitor_v4_unified.ipynb"

# We will inject a code block that temporarily overrides alpha for generation
new_source = [
    "# TEST CHECKPOINT CELL (Alpha-Dampened for Clarity)\n",
    "\n",
    "import torch\n",
    "import sys\n",
    "import os\n",
    "from transformers import AutoTokenizer\n",
    "\n",
    "# Add root to path so we can import indra\n",
    'sys.path.append(os.path.abspath(".."))\n',
    "from indra.models.quantum_model_v2 import IndraQuantumPhase2\n",
    "\n",
    "# --- CONFIG ---\n",
    'checkpoint_path = "../checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt"\n',
    "prompts = [\n",
    '    "The future of AI is",\n',
    '    "Quantum mechanics explains",\n',
    '    "### Instruction:\\nWhat is the capital of France?\\n\\n### Response:\\n",\n',
    "]\n",
    'device = "cuda" if torch.cuda.is_available() else "cpu"\n',
    "# --------------\n",
    "\n",
    "if os.path.exists(checkpoint_path):\n",
    '    print(f"Loading {checkpoint_path}...")\n',
    "    try:\n",
    "        # 1. Initialize Model\n",
    "        model = IndraQuantumPhase2(32000, 128).to(device)\n",
    "        \n",
    "        # 2. Load Checkpoint\n",
    "        try:\n",
    "            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)\n",
    "        except TypeError:\n",
    "            ckpt = torch.load(checkpoint_path, map_location=device)\n",
    "             \n",
    '        if "model_state_dict" in ckpt:\n',
    '            model.load_state_dict(ckpt["model_state_dict"])\n',
    "        else:\n",
    "            model.load_state_dict(ckpt)\n",
    "            \n",
    "        model.eval()\n",
    "\n",
    "        # 3. Check & Dampen Alpha\n",
    "        # We check the trained alpha, but use a lower one for generation to see 'metrics' clearly\n",
    "        if hasattr(model, 'output_layer') and hasattr(model.output_layer, 'alpha'):\n",
    "            trained_alpha = torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4\n",
    '            print(f"Trained Hybrid Alpha: {trained_alpha:.6f} (Too high for early inference)")\n',
    "            \n",
    "            # TEMPORARY OVERRIDE for Visualization\n",
    "            # We force alpha low so we can see the 'Real' semantic learning without Quantum Noise\n",
    "            with torch.no_grad():\n",
    "                 # Log(exp(0.05) - 1) approx -3.0. Softplus(-3.0) ~= 0.05\n",
    "                 model.output_layer.alpha.data.fill_(-3.0) \n",
    "            \n",
    "            used_alpha = torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4\n",
    '            print(f"Visualizing with Damped Alpha: {used_alpha:.6f}")\n',
    "\n",
    '        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")\n',
    "\n",
    "        # 4. Generate\n",
    "        for p in prompts:\n",
    '            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)\n',
    '            print(f"\\nPrompt: {p.strip()}")\n',
    '            print("Output: ", end="")\n',
    "\n",
    "            with torch.no_grad():\n",
    "                output_ids = input_ids.clone()\n",
    "                for _ in range(50):\n",
    "                    logits, _, _ = model(output_ids, attention_mask=None)\n",
    "                    next_token_logits = logits[:, -1, :]\n",
    "                    \n",
    "                    # Low Temperature + Sampling = Best balance for checking early progress\n",
    "                    temperature = 0.5\n",
    "                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)\n",
    "                    next_token = torch.multinomial(probs, num_samples=1)\n",
    "                    \n",
    "                    output_ids = torch.cat([output_ids, next_token], dim=1)\n",
    "                    token_str = tokenizer.decode(next_token[0])\n",
    '                    print(token_str, end="", flush=True)\n',
    "                    if next_token.item() == tokenizer.eos_token_id:\n",
    "                        break\n",
    "            print()\n",
    "\n",
    "    except Exception as e:\n",
    '        print(f"Error running model: {e}")\n',
    "else:\n",
    '    print(f"Checkpoint not found: {checkpoint_path}")',
]

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for cell in nb["cells"]:
    if cell.get("id") == "test_checkpoint":
        cell["source"] = new_source
        found = True
        break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with Alpha Dampening.")
else:
    print("Notebook cell not found.")
