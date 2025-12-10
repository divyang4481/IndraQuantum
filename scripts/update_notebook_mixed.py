import json
import os

notebook_path = "notebooks/training_monitor.ipynb"

new_cells = [
    {
        "cell_type": "markdown",
        "id": "phase2_stage1_mixed_md",
        "metadata": {},
        "source": [
            "## 6. Phase 2 (Stage 1.5): Mixed Pre-training (WikiText + Alpaca)\n",
            "\n",
            "Training on a combined dataset of General Knowledge (WikiText) and Instructions (Alpaca) to build a robust base before distillation.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "phase2_stage1_mixed_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "\n",
            "# Load Mixed Stage 1 Data\n",
            'log_file_mixed = "../logs/training_metrics_stage1_mixed.csv"\n',
            "if os.path.exists(log_file_mixed):\n",
            "    df_mixed = pd.read_csv(log_file_mixed)\n",
            "    plt.figure(figsize=(12, 6))\n",
            '    plt.plot(df_mixed["Loss"], label="Mixed Stage 1 Loss", alpha=0.4, color="gray")\n',
            '    plt.plot(df_mixed["Loss"].rolling(50).mean(), label="Smoothed (WA=50)", color="purple", linewidth=2)\n',
            '    plt.title("Phase 2 Stage 1.5: Mixed Pre-training Loss")\n',
            '    plt.xlabel("Batch")\n',
            '    plt.ylabel("Loss")\n',
            "    plt.legend()\n",
            "    plt.grid(True, alpha=0.3)\n",
            "    plt.show()\n",
            "else:\n",
            '    print("Mixed Stage 1 log not found.")',
        ],
    },
]

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Append new cells
nb["cells"].extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with Mixed Stage 1 analysis.")
