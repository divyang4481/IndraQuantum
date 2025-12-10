import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metrics():
    log_file = "logs/training_metrics_v4_unified.csv"
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if len(data) < 2:
        print("Not enough data to plot.")
        return

    # Create Steps column
    # If Batch resets every epoch, Step = (Epoch-1)*max_batch + Batch
    # Warning: If max_batch varies (e.g. last batch), this is approx.
    # Ideally find max batch from data or assume fixed.
    max_batch = data["Batch"].max()
    data["Step"] = (data["Epoch"] - 1) * max_batch + data["Batch"]

    # Smoothing
    window = 10
    if len(data) > 100:
        window = 50

    # 1. Main Loss Plot
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        data["Step"], data["Total_Loss"], label="Total Loss", alpha=0.3, color="gray"
    )
    if len(data) > window:
        plt.plot(
            data["Step"],
            data["Total_Loss"].rolling(window).mean(),
            label="Total (Smooth)",
            color="black",
        )
        plt.plot(
            data["Step"],
            data["CE"].rolling(window).mean(),
            label="CE (Grammar)",
            color="blue",
        )
        plt.plot(
            data["Step"],
            data["KD"].rolling(window).mean(),
            label="KD (Logic)",
            color="orange",
        )
    else:
        plt.plot(data["Step"], data["CE"], label="CE", color="blue")

    plt.title(f"Training Losses (Window={window})")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    if "Phase" in data.columns:
        plt.plot(
            data["Step"],
            data["Phase"].rolling(window).mean(),
            label="Phase Loss",
            color="purple",
        )
    if "Mag" in data.columns:
        plt.plot(
            data["Step"],
            data["Mag"].rolling(window).mean(),
            label="Mag Loss",
            color="green",
        )
    plt.xlabel("Global Step")
    plt.ylabel("Aux Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/loss_curves.png")
    print("Saved logs/loss_curves.png")

    # 2. Quantum Stats Plot (if available)
    if "Mag_Mean" in data.columns and len(data) > window:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        mag_mean = data["Mag_Mean"].rolling(window).mean()
        plt.plot(data["Step"], mag_mean, label="Mean Mag", color="green")
        if "Mag_Std" in data.columns:
            mag_std = data["Mag_Std"].rolling(window).mean()
            plt.fill_between(
                data["Step"],
                mag_mean - mag_std,
                mag_mean + mag_std,
                color="green",
                alpha=0.1,
            )
        plt.title("Hidden State Magnitude (Confidence)")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        phase_mean = data["Phase_Mean"].rolling(window).mean()
        plt.plot(data["Step"], phase_mean, label="Mean Phase", color="purple")
        if "Phase_Std" in data.columns:
            phase_std = data["Phase_Std"].rolling(window).mean()
            plt.fill_between(
                data["Step"],
                phase_mean - phase_std,
                phase_mean + phase_std,
                color="purple",
                alpha=0.1,
            )
        plt.title("Hidden State Phase (Entropy/Diversity)")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("logs/quantum_stats.png")
        print("Saved logs/quantum_stats.png")


if __name__ == "__main__":
    plot_metrics()
