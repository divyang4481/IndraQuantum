import json
import os

notebook_path = "notebooks/training_monitor.ipynb"

new_cells = [
    {
        "cell_type": "markdown",
        "id": "phase2_combined_loss_md",
        "metadata": {},
        "source": [
            "## 5. Phase 2: Complete Training Dynamics (Stage 1 + Stage 2)\n",
            "\n",
            "This comprehensive chart visualizes the transition from Pre-training (Stage 1) to Semantic Distillation (Stage 2).\n",
            "\n",
            "*   **Left Axis (Blue/Green):** Tracks the **Total Loss** and **Cross-Entropy (CE) Loss**. Note how CE improves consistently across stages.\n",
            "*   **Right Axis (Orange):** Tracks the **Knowledge Distillation (KD) Loss**. This appears only in Stage 2, representing the effort to mimic the Teacher model's probability distribution.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "phase2_combined_loss_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import os\n",
            "\n",
            "# Paths\n",
            'stage1_log = "../logs/training_metrics_stage1.csv"\n',
            'stage2_log = "../logs/training_metrics_stage2_v3.csv"\n',
            "\n",
            "# Data Containers\n",
            "total_loss = []\n",
            "ce_loss = []\n",
            "kd_loss = []\n",
            "stage_boundaries = []\n",
            "\n",
            "# Load Stage 1\n",
            "if os.path.exists(stage1_log):\n",
            "    df1 = pd.read_csv(stage1_log)\n",
            "    # In Stage 1, Total Loss is just CE Loss (Basic Pre-training)\n",
            '    total_loss.extend(df1["Loss"].tolist())\n',
            '    ce_loss.extend(df1["Loss"].tolist())\n',
            "    kd_loss.extend([None] * len(df1)) # No KD in Stage 1\n",
            "    stage_boundaries.append(len(total_loss))\n",
            '    print(f"Loaded Stage 1: {len(df1)} steps")\n',
            "\n",
            "# Load Stage 2\n",
            "if os.path.exists(stage2_log):\n",
            "    df2 = pd.read_csv(stage2_log)\n",
            '    total_loss.extend(df2["Total_Loss"].tolist())\n',
            '    ce_loss.extend(df2["CE"].tolist())\n',
            '    kd_loss.extend(df2["KD"].tolist())\n',
            '    print(f"Loaded Stage 2: {len(df2)} steps")\n',
            "\n",
            "# Smoothing\n",
            "window = 50\n",
            "total_smooth = pd.Series(total_loss).rolling(window).mean()\n",
            "ce_smooth = pd.Series(ce_loss).rolling(window).mean()\n",
            "kd_smooth = pd.Series(kd_loss).rolling(window).mean()\n",
            "\n",
            "# Plotting\n",
            "fig, ax1 = plt.subplots(figsize=(16, 8))\n",
            "\n",
            "# Axis 1: Total Loss & CE (Left Scale)\n",
            'ax1.plot(total_smooth, label="Total Loss", color="blue", linewidth=2, alpha=0.8)\n',
            'ax1.plot(ce_smooth, label="CE Loss (Prediction)", color="green", linewidth=2, linestyle="--", alpha=0.8)\n',
            'ax1.set_xlabel("Global Batch Step")\n',
            'ax1.set_ylabel("Total / CE Loss", color="blue")\n',
            "ax1.tick_params(axis='y', labelcolor=\"blue\")\n",
            "ax1.grid(True, alpha=0.3)\n",
            "\n",
            "# Axis 2: KD Loss (Right Scale)\n",
            "ax2 = ax1.twinx()\n",
            'ax2.plot(kd_smooth, label="KD Loss (Teacher Matching)", color="orange", linewidth=2, linestyle=":", alpha=0.9)\n',
            'ax2.set_ylabel("KD Loss (Raw)", color="orange")\n',
            "ax2.tick_params(axis='y', labelcolor=\"orange\")\n",
            "\n",
            "# Stage Markers\n",
            "for boundary in stage_boundaries:\n",
            '    ax1.axvline(x=boundary, color="red", linestyle="-", linewidth=2, alpha=0.5)\n',
            '    ax1.text(boundary + 50, ax1.get_ylim()[1]*0.95, "Stage 2 Starts (Distillation)", color="red", fontweight="bold")\n',
            "\n",
            "# Title & Legend\n",
            'plt.title("Phase 2 Training Dynamics: Evolution of Learning & Distillation")\n',
            "lines1, labels1 = ax1.get_legend_handles_labels()\n",
            "lines2, labels2 = ax2.get_legend_handles_labels()\n",
            'ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)\n',
            "\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ],
    },
]

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Modify the LAST cell if it's already the combined one, or Append if not
# We will just append for safety, user can delete duplicates.
nb["cells"].extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with Advanced Combined Loss Curve.")
