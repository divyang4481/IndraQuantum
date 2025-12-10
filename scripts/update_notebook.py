import json
import os

notebook_path = "notebooks/monitor_v4_unified.ipynb"

new_source = [
    "# TEST CHECKPOINT CELL\n",
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
    "        # Initialize Model (Masking fixed architecture)\n",
    "        model = IndraQuantumPhase2(32000, 128).to(device)\n",
    "        \n",
    "        # Load with weights_only=False because custom classes might be pickled\n",
    "        try:\n",
    "            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)\n",
    "        except TypeError:\n",
    "             # Fallback for older torch versions or if param not supported in this context\n",
    "             ckpt = torch.load(checkpoint_path, map_location=device)\n",
    "             \n",
    '        if "model_state_dict" in ckpt:\n',
    '            model.load_state_dict(ckpt["model_state_dict"])\n',
    "        else:\n",
    "            model.load_state_dict(ckpt)\n",
    "            \n",
    "        model.eval()\n",
    "\n",
    "        # Check alpha value\n",
    "        if hasattr(model, 'output_layer') and hasattr(model.output_layer, 'alpha'):\n",
    "            alpha_val = torch.nn.functional.softplus(model.output_layer.alpha).item() + 1e-4\n",
    '            print(f"Current Hybrid Alpha: {alpha_val:.6f}")\n',
    "\n",
    '        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")\n',
    "\n",
    "        for p in prompts:\n",
    '            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)\n',
    '            print(f"\\nPrompt: {p.strip()}")\n',
    '            print("Output: ", end="")\n',
    "\n",
    "            # Generate with Sampling\n",
    "            with torch.no_grad():\n",
    "                output_ids = input_ids.clone()\n",
    "                for _ in range(50):\n",
    "                    # pass Mask=None for inference\n",
    "                    logits, _, _ = model(output_ids, attention_mask=None)\n",
    "                    next_token_logits = logits[:, -1, :]\n",
    "                    \n",
    "                    # Repetition Penalty\n",
    "                    repetition_penalty = 1.2\n",
    "                    for token_id in set(output_ids[0].tolist()):\n",
    "                        if next_token_logits[0, token_id] < 0:\n",
    "                            next_token_logits[0, token_id] *= repetition_penalty\n",
    "                        else:\n",
    "                            next_token_logits[0, token_id] /= repetition_penalty\n",
    "                    \n",
    "                    # Apply Temperature\n",
    "                    temperature = 0.7\n",
    "                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)\n",
    "                    \n",
    "                    # Sample\n",
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
    print("Notebook updated successfully.")
else:
    print("Cell with id 'test_checkpoint' not found.")
