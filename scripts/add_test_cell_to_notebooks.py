import json
import os

# Update Stage 1 Mixed Notebook
nb1_path = "notebooks/monitor_stage1_mixed.ipynb"

if os.path.exists(nb1_path):
    with open(nb1_path, "r", encoding="utf-8") as f:
        nb1 = json.load(f)

    test_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "test_checkpoint",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test Checkpoint Cell\n",
            "import torch\n",
            "import sys\n",
            "import os\n",
            'sys.path.append(os.path.abspath(".."))\n',
            "from indra.models.quantum_model_v2 import IndraQuantumPhase2\n",
            "from transformers import AutoTokenizer\n",
            "\n",
            "# --- Config ---\n",
            'checkpoint_path = "../checkpoints/phase2_stage1_mixed/checkpoint_stage1_mixed_epoch_1.pt" # Change Epoch here\n',
            "prompts = [\n",
            '    "The future of AI is",\n',
            '    "Once upon a time",\n',
            '    "Quantum physics explains"\n',
            "]\n",
            'device = "cuda" if torch.cuda.is_available() else "cpu"\n',
            "# ----------------\n",
            "\n",
            "if os.path.exists(checkpoint_path):\n",
            '    print(f"Loading {checkpoint_path}...")\n',
            "    model = IndraQuantumPhase2(32000, 128).to(device)\n",
            "    ckpt = torch.load(checkpoint_path, map_location=device)\n",
            '    model.load_state_dict(ckpt["model_state_dict"])\n',
            "    model.eval()\n",
            "    \n",
            '    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")\n',
            "    \n",
            "    for p in prompts:\n",
            '        input_ids = tokenizer.encode(p, return_tensors="pt").to(device)\n',
            '        print(f"\\nPrompt: {p}")\n',
            '        print("Output: ", end="")\n',
            "        \n",
            "        # Simple Greedy Gen\n",
            "        with torch.no_grad():\n",
            "            output_ids = input_ids.clone()\n",
            "            for _ in range(50):\n",
            "                logits, _, _ = model(output_ids)\n",
            "                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n",
            "                output_ids = torch.cat([output_ids, next_token], dim=1)\n",
            '                print(tokenizer.decode(next_token[0]), end="", flush=True)\n',
            "                if next_token.item() == tokenizer.eos_token_id:\n",
            "                    break\n",
            "        print()\n",
            "else:\n",
            '    print(f"Checkpoint not found: {checkpoint_path}")',
        ],
    }

    nb1["cells"].append(test_cell)

    with open(nb1_path, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1)
    print(f"Updated {nb1_path}")


# Update v4 Unified Notebook
nb2_path = "notebooks/monitor_v4_unified.ipynb"

if os.path.exists(nb2_path):
    with open(nb2_path, "r", encoding="utf-8") as f:
        nb2 = json.load(f)

    test_cell_v4 = {
        "cell_type": "code",
        "execution_count": None,
        "id": "test_checkpoint_v4",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test Checkpoint Cell (v4)\n",
            "import torch\n",
            "import sys\n",
            "import os\n",
            'sys.path.append(os.path.abspath(".."))\n',
            "from indra.models.quantum_model_v2 import IndraQuantumPhase2\n",
            "from transformers import AutoTokenizer\n",
            "\n",
            "# --- Config ---\n",
            'checkpoint_path = "../checkpoints/phase2_v4_unified/checkpoint_v4_epoch_1.pt" # Change Epoch here\n',
            "prompts = [\n",
            '    "The future of AI is",\n',
            '    "### Instruction:\\nWhat is quantum mechanics?\\n\\n### Response:\\n"\n',
            "]\n",
            'device = "cuda" if torch.cuda.is_available() else "cpu"\n',
            "# ----------------\n",
            "\n",
            "if os.path.exists(checkpoint_path):\n",
            '    print(f"Loading {checkpoint_path}...")\n',
            "    model = IndraQuantumPhase2(32000, 128).to(device)\n",
            "    ckpt = torch.load(checkpoint_path, map_location=device)\n",
            '    model.load_state_dict(ckpt["model_state_dict"])\n',
            "    model.eval()\n",
            "    \n",
            '    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")\n',
            "    \n",
            "    for p in prompts:\n",
            '        input_ids = tokenizer.encode(p, return_tensors="pt").to(device)\n',
            '        print(f"\\nPrompt: {p}")\n',
            '        print("Output: ", end="")\n',
            "        \n",
            "        # Simple Greedy Gen\n",
            "        with torch.no_grad():\n",
            "            output_ids = input_ids.clone()\n",
            "            for _ in range(50):\n",
            "                logits, _, _ = model(output_ids)\n",
            "                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n",
            "                output_ids = torch.cat([output_ids, next_token], dim=1)\n",
            '                print(tokenizer.decode(next_token[0]), end="", flush=True)\n',
            "                if next_token.item() == tokenizer.eos_token_id:\n",
            "                    break\n",
            "        print()\n",
            "else:\n",
            '    print(f"Checkpoint not found: {checkpoint_path}")',
        ],
    }

    nb2["cells"].append(test_cell_v4)

    with open(nb2_path, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=1)
    print(f"Updated {nb2_path}")
