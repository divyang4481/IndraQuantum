import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_advanced_metrics():
    log_file = "logs/training_metrics_rank64_optimized.csv"
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    try:
        df = pd.read_csv(log_file)

        # Handle column naming
        if "KD_Logits" in df.columns:
            df["KD_Loss"] = df["KD_Logits"]

        # Group by Epoch for trend lines
        epoch_data = df.groupby("Epoch").mean()

        # Calculate Perplexity (e^(CE Loss))
        epoch_data["Perplexity"] = np.exp(epoch_data["CE_Loss"])

        # Create a Grid of Plots
        fig = plt.figure(figsize=(18, 12))

        # 1. Perplexity Curve (The "Language Understanding" Metric)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(
            epoch_data.index,
            epoch_data["Perplexity"],
            color="purple",
            linewidth=2,
            marker="o",
        )
        ax1.set_title("Model Perplexity (Lower is Better)")
        ax1.set_ylabel("Perplexity (e^CE)")
        ax1.set_xlabel("Epoch")
        ax1.grid(True, alpha=0.3)
        # Interpretive text
        last_ppl = epoch_data["Perplexity"].iloc[-1]
        ax1.text(epoch_data.index[-1], last_ppl, f"{last_ppl:.1f}", fontweight="bold")

        # 2. Loss Landscape Overlay (Total vs KD vs CE) normalized
        ax2 = fig.add_subplot(2, 2, 2)
        # Normalize to start at 1.0 to compare rates of decent
        norm_total = epoch_data["Total_Loss"] / epoch_data["Total_Loss"].iloc[0]
        norm_kd = epoch_data["KD_Loss"] / epoch_data["KD_Loss"].iloc[0]
        norm_ce = epoch_data["CE_Loss"] / epoch_data["CE_Loss"].iloc[0]

        ax2.plot(epoch_data.index, norm_total, label="Total Loss", linestyle="-")
        ax2.plot(
            epoch_data.index,
            norm_kd,
            label="KD Loss (Teacher Matching)",
            linestyle="--",
        )
        ax2.plot(epoch_data.index, norm_ce, label="CE Loss (Truth)", linestyle=":")
        ax2.set_title("Relative Improvement Rate (Normalized to 1.0)")
        ax2.set_ylabel("Relative Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 3D Trajectory of Learning
        # X=Epoch, Y=CE, Z=KD
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        x = epoch_data.index
        y = epoch_data["CE_Loss"]
        z = epoch_data["KD_Loss"]

        ax3.plot(x, y, z, color="teal", linewidth=2)
        ax3.scatter(x, y, z, c=x, cmap="viridis")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("CE Loss (Accuracy)")
        ax3.set_zlabel("KD Loss (Reasoning)")
        ax3.set_title("3D Learning Trajectory")

        # 4. Entropy Proxy (Variance of Batch Loss)
        # Can't calculate true entropy without logits, but batch variance
        # is a proxy for "Confidence Stability"
        # We calculate variance of Total_Loss within each epoch
        epoch_var = df.groupby("Epoch")["Total_Loss"].var()
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(epoch_var.index, epoch_var, color="orange", linewidth=2)
        ax4.set_title("Stability Metric (Loss Variance)")
        ax4.set_ylabel("Batch-to-Batch Variance")
        ax4.set_xlabel("Epoch")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = "plots/advanced_analysis.png"
        plt.savefig(output_path)
        print(f"Advanced analysis saved to {output_path}")

    except Exception as e:
        print(f"Error plotting advanced analysis: {e}")


if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_advanced_metrics()
