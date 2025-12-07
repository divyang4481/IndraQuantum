import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_batch_oscillation():
    log_file = "logs/training_metrics_rank64_optimized.csv"
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    try:
        df = pd.read_csv(log_file)

        # Handle column naming
        if "KD_Logits" in df.columns:
            df["KD_Loss"] = df["KD_Logits"]

        # Use the whole history to show the full "curve with variance"
        # Create 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        step = df.index
        window_size = 50  # Rolling window for trend

        # 1. Total Loss
        ax1.plot(
            step,
            df["Total_Loss"],
            label="Raw Batch",
            color="gray",
            alpha=0.3,
            linewidth=0.5,
        )
        ax1.plot(
            step,
            df["Total_Loss"].rolling(window_size).mean(),
            label=f"Trend ({window_size} avg)",
            color="blue",
            linewidth=2,
        )
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Total Loss: Noise vs Trend")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # 2. CE Loss (Accuracy)
        ax2.plot(
            step,
            df["CE_Loss"],
            label="Raw Batch",
            color="pink",
            alpha=0.4,
            linewidth=0.5,
        )
        ax2.plot(
            step,
            df["CE_Loss"].rolling(window_size).mean(),
            label=f"Trend ({window_size} avg)",
            color="red",
            linewidth=2,
        )
        ax2.set_ylabel("CE Loss")
        ax2.set_title("Cross Entropy (Accuracy): Noise vs Trend")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # 3. KD Loss (Reasoning)
        ax3.plot(
            step,
            df["KD_Loss"],
            label="Raw Batch",
            color="lightgreen",
            alpha=0.4,
            linewidth=0.5,
        )
        ax3.plot(
            step,
            df["KD_Loss"].rolling(window_size).mean(),
            label=f"Trend ({window_size} avg)",
            color="darkgreen",
            linewidth=2,
        )
        ax3.set_ylabel("KD Loss")
        ax3.set_title("Distillation (KD): Noise vs Trend")
        ax3.set_xlabel("Training Steps (Batches)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = "plots/batch_oscillation_full.png"
        plt.savefig(output_path)
        print(f"Full batch oscillation plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting batch oscillation: {e}")


def plot_optimized_run():
    log_file = "logs/training_metrics_rank64_optimized.csv"

    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    try:
        df = pd.read_csv(log_file)

        # Handle column naming if needed
        if "KD_Logits" in df.columns:
            df["KD_Loss"] = df["KD_Logits"]

        # Group by Epoch for smoother lines
        epoch_data = df.groupby("Epoch").mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Total Loss & KD Loss
        ax1.plot(
            epoch_data.index,
            epoch_data["Total_Loss"],
            label="Total Loss",
            color="blue",
            linewidth=2,
        )
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Optimized Training Progress (Rank 64, T=1.0, Low KD)")
        ax1.grid(True, alpha=0.3)

        # Add KD on secondary axis
        ax1_b = ax1.twinx()
        ax1_b.plot(
            epoch_data.index,
            epoch_data["KD_Loss"],
            label="KD Loss (Raw)",
            color="green",
            linestyle="--",
            alpha=0.5,
        )
        ax1_b.set_ylabel("KD Loss Magnitude")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Plot 2: Cross Entropy (Accuracy)
        ax2.plot(
            epoch_data.index,
            epoch_data["CE_Loss"],
            label="CE Loss (Accuracy)",
            color="red",
            linewidth=2,
        )
        ax2.set_ylabel("Cross Entropy Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Prediction Accuracy (Lower is Better)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = "plots/loss_curve_optimized.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting: {e}")


def plot_epoch_oscillation():
    log_file = "logs/training_metrics_rank64_optimized.csv"
    if not os.path.exists(log_file):
        return

    try:
        df = pd.read_csv(log_file)

        # Calculate stats per epoch
        epoch_stats = df.groupby("Epoch").agg(
            {
                "Total_Loss": ["mean", "std"],
                "CE_Loss": ["mean", "std", "min", "max"],
                "KD_Logits": (
                    ["mean", "std"] if "KD_Logits" in df.columns else ["mean", "std"]
                ),
            }
        )

        epoch_stats.columns = [
            "_".join(col).strip() for col in epoch_stats.columns.values
        ]
        epochs = epoch_stats.index

        fig, ax = plt.subplots(figsize=(12, 8))

        # Main Curve (CE MEAN)
        ax.plot(
            epochs,
            epoch_stats["CE_Loss_mean"],
            label="Average CE Loss",
            color="red",
            linewidth=3,
        )

        # Oscillation (Std Dev)
        ax.fill_between(
            epochs,
            epoch_stats["CE_Loss_mean"] - epoch_stats["CE_Loss_std"],
            epoch_stats["CE_Loss_mean"] + epoch_stats["CE_Loss_std"],
            color="red",
            alpha=0.2,
            label="Oscillation (±1 Std Dev)",
        )

        # Min/Max Range
        ax.fill_between(
            epochs,
            epoch_stats["CE_Loss_min"],
            epoch_stats["CE_Loss_max"],
            color="gray",
            alpha=0.1,
            label="Full CE Range (Min-Max)",
        )

        ax.set_title("Training Stability: Cross-Entropy (Accuracy) Oscillation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        curr_mean = epoch_stats["CE_Loss_mean"].iloc[-1]
        curr_std = epoch_stats["CE_Loss_std"].iloc[-1]
        ax.text(
            epochs[-1],
            curr_mean,
            f" {curr_mean:.2f} (±{curr_std:.2f})",
            fontweight="bold",
            color="red",
        )

        output_path = "plots/epoch_oscillation.png"
        plt.savefig(output_path)
        print(f"Epoch oscillation plot saved to {output_path}")

    except Exception as e:
        print(f"Error plotting epoch oscillation: {e}")


if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_optimized_run()
    plot_batch_oscillation()
    plot_epoch_oscillation()
