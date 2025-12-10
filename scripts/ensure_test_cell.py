import json
import os

nb_path = "notebooks/monitor_v4_unified.ipynb"

if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if test cell already exists to avoid duplicates
    has_test_cell = False
    for cell in nb["cells"]:
        if cell["source"] and "# Test Checkpoint Cell" in cell["source"][0]:
            has_test_cell = True
            break

    if not has_test_cell:
        test_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "test_checkpoint_v4_new",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test Checkpoint Cell (Interactive)\n",
                "import torch\n",
                "import sys\n",
                "import os\n",
                'sys.path.append(os.path.abspath(".."))\n',
                "from indra.models.quantum_model_v2 import IndraQuantumPhase2\n",
                "from transformers import AutoTokenizer\n",
                "\n",
                "# --- Configuration ---\n",
                'checkpoint_path = "../checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt" # UPDATE THIS EPOCH NUMBER\n',
                "prompts = [\n",
                '    "The future of AI is",\n',
                '    "### Instruction:\\nWhat is quantum mechanics?\\n\\n### Response:\\n",\n',
                '    "Once upon a time in a land far away,"\n',
                "]\n",
                'device = "cuda" if torch.cuda.is_available() else "cpu"\n',
                "# ---------------------\n",
                "\n",
                "if os.path.exists(checkpoint_path):\n",
                '    print(f"Loading {checkpoint_path}...")\n',
                "    try:\n",
                "        model = IndraQuantumPhase2(32000, 128).to(device)\n",
                "        ckpt = torch.load(checkpoint_path, map_location=device)\n",
                '        model.load_state_dict(ckpt["model_state_dict"])\n',
                "        model.eval()\n",
                "        \n",
                '        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")\n',
                "        \n",
                "        for p in prompts:\n",
                '            input_ids = tokenizer.encode(p, return_tensors="pt").to(device)\n',
                '            print(f"\\nPrompt: {p.strip()}")\n',
                '            print("Output: ", end="")\n',
                "            \n",
                "            # Greedy Generation Loop\n",
                "            with torch.no_grad():\n",
                "                output_ids = input_ids.clone()\n",
                "                for _ in range(50):\n",
                "                    # Note: We pass None for mask during inference for simple greedy generation\n",
                "                    # (Model should handle unmasked self-attention on its own history)\n",
                "                    logits, _, _ = model(output_ids)\n",
                "                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n",
                "                    output_ids = torch.cat([output_ids, next_token], dim=1)\n",
                "                    token_str = tokenizer.decode(next_token[0])\n",
                '                    print(token_str, end="", flush=True)\n',
                "                    if next_token.item() == tokenizer.eos_token_id:\n",
                "                        break\n",
                "            print()\n",
                "            \n",
                "    except Exception as e:\n",
                '        print(f"Error loading/running model: {e}")\n',
                "else:\n",
                '    print(f"Checkpoint not found: {checkpoint_path}\\nWait for Epoch 1 to finish!")',
            ],
        }

        nb["cells"].append(test_cell)

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Added test cell to {nb_path}")
    else:
        print("Test cell already exists in notebook.")
