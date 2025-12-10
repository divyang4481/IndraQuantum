import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Config
log_file = "logs/training_metrics_v4_unified.csv"
window = 20  # Smoother window for high-res analysis


def plot_metrics():
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            print(f"Loaded {len(df)} steps.")

            if len(df) < window:
                print(
                    f"Not enough data points ({len(df)}) for rolling window of {window}. Showing raw data."
                )
                window_curr = 1
            else:
                window_curr = window

            # Add rolling averages
            df["Mag_Smooth"] = df["Mag"].rolling(window_curr, min_periods=1).mean()
            df["Phase_Smooth"] = df["Phase"].rolling(window_curr, min_periods=1).mean()
            df["CE_Smooth"] = df["CE"].rolling(window_curr, min_periods=1).mean()
            df["Batch_Index"] = df.index

            # --- PLOT 1: STANDARD LOSS CURVES ---\n
            fig, ax1 = plt.subplots(figsize=(15, 6))
            ax1.plot(
                df["CE_Smooth"], label="CE Loss (Accuracy)", color="green", linewidth=2
            )
            ax1.plot(
                df["Mag_Smooth"],
                label="Mag Loss (Confidence)",
                color="blue",
                linestyle="--",
            )
            ax1.set_xlabel("Training Steps (x10 Batches)")
            ax1.set_ylabel("CE / Mag Loss")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                df["Phase_Smooth"],
                label="Phase Loss (Structure)",
                color="red",
                linestyle=":",
                linewidth=2,
            )
            ax2.set_ylabel("Phase Loss", color="red")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            plt.title("Training Dynamics: Accuracy vs Quantum Structure")
            plt.savefig("logs/training_dynamics_plot.png")
            print("Saved logs/training_dynamics_plot.png")
            plt.show()

            # --- PLOT 2: PHASE-MAG CORRELATION (2D SCATTER) ---\n
            # Does high confidence (Low Mag Loss) correlate with better structure (Low Phase Loss)?
            plt.figure(figsize=(10, 8))
            sc = plt.scatter(
                df["Mag_Smooth"],
                df["Phase_Smooth"],
                c=df["Batch_Index"],
                cmap="viridis",
                alpha=0.6,
                s=10,
            )
            plt.colorbar(sc, label="Training Progress (Steps)")
            plt.xlabel("Magnitude Loss (Salience/Confidence)")
            plt.ylabel("Phase Loss (Context/Uncertainty)")
            plt.title("Quantum Optimization Path: Mag vs Phase")
            plt.grid(True, alpha=0.3)
            # Arrow to show direction
            if len(df) > 10:
                plt.arrow(
                    df["Mag_Smooth"].iloc[0],
                    df["Phase_Smooth"].iloc[0],
                    df["Mag_Smooth"].iloc[-1] - df["Mag_Smooth"].iloc[0],
                    df["Phase_Smooth"].iloc[-1] - df["Phase_Smooth"].iloc[0],
                    head_width=0.05,
                    color="red",
                    length_includes_head=True,
                )
                plt.text(
                    df["Mag_Smooth"].iloc[0],
                    df["Phase_Smooth"].iloc[0],
                    "Start",
                    color="red",
                    fontweight="bold",
                )
                plt.text(
                    df["Mag_Smooth"].iloc[-1],
                    df["Phase_Smooth"].iloc[-1],
                    "Current",
                    color="red",
                    fontweight="bold",
                )
            plt.savefig("logs/phase_mag_correlation.png")
            print("Saved logs/phase_mag_correlation.png")
            plt.show()

            # --- PLOT 3: 3D LEARNING TRAJECTORY ---\n
            # Visualizing the descent in the 3D Loss Landscape
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot trajectory
            p = ax.scatter(
                df["Batch_Index"],
                df["Mag_Smooth"],
                df["Phase_Smooth"],
                c=df["CE_Smooth"],
                cmap="coolwarm",
                s=20,
            )

            ax.set_xlabel("Time (Steps)")
            ax.set_ylabel("Mag Loss")
            ax.set_zlabel("Phase Loss")
            fig.colorbar(p, label="CE Loss (Red=High, Blue=Low)")
            plt.title("3D Learning Trajectory: Time x Mag x Phase")
            plt.savefig("logs/3d_trajectory.png")
            print("Saved logs/3d_trajectory.png")
            plt.show()

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Log file not found yet.")


if __name__ == "__main__":
    plot_metrics()
