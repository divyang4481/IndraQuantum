import nbformat as nbf
import os
from datetime import datetime


def create_advanced_monitor_notebook():
    nb = nbf.v4.new_notebook()

    # 1. Title & Header
    header = nbf.v4.new_markdown_cell(
        f"""
# 🧠 IndraQuantum Training Monitor (Rank 64)
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

This notebook visualizes the training dynamics of the Quantum-Inspired Student Model.
**Goal:** Achieve stable convergence with <1M parameters using Knowledge Distillation.
    """
    )
    nb.cells.append(header)

    # 2. Load Data Block
    load_code = nbf.v4.new_code_cell(
        """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit

# Configuration
log_file = "../logs/training_metrics_stage2.csv"
if not os.path.exists(log_file):
    print(f"Log file {log_file} not found! Ensure Phase 2 Stage 2 training is running.")
    # Fallback for dev if path is different in notebook relative
    if os.path.exists("logs/training_metrics_stage2.csv"):
        log_file = "logs/training_metrics_stage2.csv"
else:
    df = pd.read_csv(log_file)
    print(f"Loaded {len(df)} batch records")
    print(f"Current Epoch: {df['Epoch'].max()}")
    print("Columns:", df.columns.tolist())
    df.tail()
"""
    )
    nb.cells.append(load_code)

    # 3. Epoch Oscillation Plot
    osc_md = nbf.v4.new_markdown_cell(
        """
## 1. Epoch Oscillation Analysis
This plot visualizes the **Stability** of learning.
- **Line**: Average CE Loss (Accuracy)
- **Shaded Area**: Standard Deviation (Batch-to-Batch Noise)
    """
    )
    osc_code = nbf.v4.new_code_cell(
        """
plt.figure(figsize=(12, 6))
# Calculate detailed stats
epoch_stats = df.groupby("Epoch")["CE_Loss"].agg(["mean", "std", "min", "max"])

# 1. Main Trend (Mean)
plt.plot(epoch_stats.index, epoch_stats["mean"], label="Mean CE Loss", color="red", linewidth=2)

# 2. Typical Oscillation (Standard Deviation)
plt.fill_between(epoch_stats.index, 
                 epoch_stats["mean"] - epoch_stats["std"], 
                 epoch_stats["mean"] + epoch_stats["std"], 
                 color="red", alpha=0.2, label="Typical Oscillation (±1σ)")

# 3. Extreme Oscillation (Min-Max Range)
plt.fill_between(epoch_stats.index, 
                 epoch_stats["min"], 
                 epoch_stats["max"], 
                 color="gray", alpha=0.1, label="Full Range (Min-Max)")

# Annotate current values
last_ep = epoch_stats.index[-1]
last_mean = epoch_stats["mean"].iloc[-1]
last_min = epoch_stats["min"].iloc[-1]
last_max = epoch_stats["max"].iloc[-1]

plt.text(last_ep, last_mean, f" Avg:{last_mean:.2f}", color='red', fontweight='bold')
plt.text(last_ep, last_max, f" Max:{last_max:.2f}", color='gray', fontsize=8)
plt.text(last_ep, last_min, f" Min:{last_min:.2f}", color='gray', fontsize=8)

plt.title("Epoch Oscillation: Mean, Std Dev, and Full Min-Max Range")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
"""
    )
    nb.cells.append(osc_md)
    nb.cells.append(osc_code)

    # Advanced 2x2 Grid (Updated for Phase 2)
    adv_md = nbf.v4.new_markdown_cell(
        """
## 2. Phase 2 Multi-Metric Analysis (Complex Architecture)
- **Top Left**: Perplexity ($e^{CE}$) - General Language Capability
- **Top Right**: Phase & Magnitude Loss - The "V2" Components (Active after Epoch 5)
- **Bottom Left**: Knowledge Distillation - Imitation Divergence
- **Bottom Right**: Learning Velocity - Stability Metric
    """
    )
    adv_code = nbf.v4.new_code_cell(
        """
fig = plt.figure(figsize=(18, 12))
epoch_mean = df.groupby("Epoch").mean()

# A. Perplexity
ax1 = fig.add_subplot(2, 2, 1)
ppl = np.exp(epoch_mean["CE"])
ax1.plot(ppl.index, ppl, color='purple', marker='o', label="Perplexity")
ax1.set_title("Perplexity (Understanding Score)")
ax1.set_ylabel("PPL")
ax1.grid(True, alpha=0.3)
ax1.legend()

# B. Phase 2 Specifics (Phase/Mag)
ax2 = fig.add_subplot(2, 2, 2)
if "Phase" in epoch_mean.columns and "Mag" in epoch_mean.columns:
    ax2.plot(epoch_mean.index, epoch_mean["Phase"], label="Phase Loss (Certainty)", color="orange")
    ax2.plot(epoch_mean.index, epoch_mean["Mag"], label="Mag Loss (Salience)", color="cyan")
    ax2.set_title("Complex Geometry Losses (Active Epoch 5+)")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Mark activation epoch
    ax2.axvline(5, color='gray', linestyle='--', alpha=0.5, label="Warmup End")
else:
    ax2.text(0.5, 0.5, "Phase/Mag data not found in logs", ha='center')

# C. KD Trend
ax3 = fig.add_subplot(2, 2, 3)
if "KD" in epoch_mean.columns:
    ax3.plot(epoch_mean.index, epoch_mean["KD"], color='green', label="KD Loss")
    ax3.set_title("Knowledge Distillation (Teacher Imitation)")
    ax3.set_ylabel("KL Div")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

# D. Learning Rate Efficiency
ax4 = fig.add_subplot(2, 2, 4)
if "Total_Loss" in epoch_mean.columns:
    loss_diff = epoch_mean["Total_Loss"].diff().fillna(0)
    ax4.bar(loss_diff.index, loss_diff, color='blue', alpha=0.6)
    ax4.set_title("Learning Velocity (Loss Delta per Epoch)")
    ax4.set_ylabel("Change in Loss")
    ax4.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.show()
"""
    )

    nb.cells.append(adv_md)
    nb.cells.append(adv_code)

    # 5. Convergence Projection (Target CE)
    conv_md = nbf.v4.new_markdown_cell(
        """
## 3. Convergence Projection
- **Target CE (Safe Goal)**: 4.0 (Typical for <1M params)
- **Target CE (Stretch Goal)**: 3.5 (Fluent English)

We estimate the **Expected Completion Epoch** based on the current learning speed (last 5 epochs).
    """
    )
    conv_code = nbf.v4.new_code_cell(
        """
TARGET_CE = 4.0
recent_window = 10 # Use trend of last 10 epochs

ce_history = df.groupby("Epoch")["CE_Loss"].mean()

if len(ce_history) > recent_window:
    # Linear fit prediction
    y_recent = ce_history.iloc[-recent_window:]
    x_recent = y_recent.index
    
    coef = np.polyfit(x_recent, y_recent, 1)
    slope = coef[0]
    intercept = coef[1]
    
    current_ce = ce_history.iloc[-1]
    dist_to_target = current_ce - TARGET_CE
    
    print(f"Current CE: {current_ce:.4f}")
    print(f"Target CE: {TARGET_CE}")
    print(f"Distance to Target: {dist_to_target:.4f}")
    print(f"Current Velocity: {slope:.4f} loss/epoch")
    
    if slope < 0 and dist_to_target > 0:
        target_epoch = (TARGET_CE - intercept) / slope
        print(f"ESTIMATED COMPLETION: Epoch {int(target_epoch)}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(ce_history.index, ce_history, label="Actual History", marker='.', color='blue')
        
        fut_x = np.linspace(ce_history.index[-1], target_epoch + 5, 20)
        fut_y = slope * fut_x + intercept
        plt.plot(fut_x, fut_y, label="Projection", linestyle="--", color="orange")
        
        plt.axhline(TARGET_CE, color="green", label="Target 4.0", linewidth=2)
        plt.scatter([target_epoch], [TARGET_CE], color="red", s=100, zorder=5)
        plt.text(target_epoch, TARGET_CE - 0.05, f" Ep {int(target_epoch)}", color="red", fontweight="bold")
        
        plt.title(f"Convergence Forecast: Reaching CE {TARGET_CE}")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Model is not converging linearly or already passed target.")
else:
    print("Need more data for projection.")
"""
    )
    nb.cells.append(conv_md)
    nb.cells.append(conv_code)

    # 6. Save
    output_path = "notebooks/training_monitor.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook generated at: {output_path}")


if __name__ == "__main__":
    create_advanced_monitor_notebook()
