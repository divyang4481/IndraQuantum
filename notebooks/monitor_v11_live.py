import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# File to monitor
csv_file = r"c:\workspace\AI\IndraQuantum\runs\IndraV11-GPUDistill-Llama_20251219-130308\metrics.csv"


def animate(i):
    # Load Data
    try:
        df = pd.read_csv(csv_file)
    except:
        return

    # Clear previous
    plt.cla()

    # Setup nice style
    plt.style.use("ggplot")

    # Data
    steps = df["step"]
    total_loss = df["total_loss"]
    ce_loss = df["ce_loss"]
    kd_loss = df["kd_loss"]  # Added KD

    # Smoothing
    smooth_window = 20
    total_smooth = total_loss.rolling(window=smooth_window, min_periods=1).mean()
    ce_smooth = ce_loss.rolling(window=smooth_window, min_periods=1).mean()
    kd_smooth = kd_loss.rolling(window=smooth_window, min_periods=1).mean()  # Added KD

    # --- PLOTTING ---

    # 1. TOTAL LOSS
    plt.plot(
        steps, total_loss, color="Salmon", alpha=0.3, label="Raw Total", linewidth=0.8
    )
    plt.plot(steps, total_smooth, color="FireBrick", linewidth=2.5, label="Total Trend")

    # 2. CE LOSS
    plt.plot(
        steps, ce_loss, color="LimeGreen", alpha=0.3, label="Raw CE", linewidth=0.8
    )
    plt.plot(
        steps,
        ce_smooth,
        color="DarkGreen",
        linewidth=2.5,
        linestyle="--",
        label="CE Trend",
    )

    # 3. KD LOSS (NEW)
    plt.plot(steps, kd_loss, color="SkyBlue", alpha=0.3, label="Raw KD", linewidth=0.8)
    plt.plot(
        steps,
        kd_smooth,
        color="RoyalBlue",
        linewidth=2.5,
        linestyle="-.",
        label="KD Trend",
    )

    # --- HIGHLIGHT: SMOOTH MINIMUMS (STARS) ---

    # Total Smooth
    min_total_smooth_idx = total_smooth.idxmin()
    val_total_smooth = total_smooth[min_total_smooth_idx]
    step_total_smooth = steps[min_total_smooth_idx]

    plt.scatter(
        [step_total_smooth],
        [val_total_smooth],
        color="black",
        edgecolor="DarkRed",
        marker="*",
        s=200,
        linewidth=2,
        zorder=6,
        # label="Deepest Smooth Total",
    )

    # CE Smooth
    min_ce_smooth_idx = ce_smooth.idxmin()
    val_ce_smooth = ce_smooth[min_ce_smooth_idx]
    step_ce_smooth = steps[min_ce_smooth_idx]

    plt.scatter(
        [step_ce_smooth],
        [val_ce_smooth],
        color="black",
        edgecolor="DarkGreen",
        marker="*",
        s=200,
        linewidth=2,
        zorder=6,
        # label="Deepest Smooth CE",
    )

    # KD Smooth
    min_kd_smooth_idx = kd_smooth.idxmin()
    val_kd_smooth = kd_smooth[min_kd_smooth_idx]
    step_kd_smooth = steps[min_kd_smooth_idx]

    plt.scatter(
        [step_kd_smooth],
        [val_kd_smooth],
        color="black",
        edgecolor="RoyalBlue",
        marker="*",
        s=200,
        linewidth=2,
        zorder=6,
        # label="Deepest Smooth KD",
    )

    # --- DASHBOARD BOX ---
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    stats_text = (
        f"LATEST (Step {steps.iloc[-1]}):\n"
        f"--------------------\n"
        f"Total: {total_smooth.iloc[-1]:.2f}\n"
        f"CE:    {ce_smooth.iloc[-1]:.2f}\n"
        f"KD:    {kd_smooth.iloc[-1]:.2f}\n"
        f"--------------------\n"
        f"RECORDS (Smooth Min):\n"
        f"Total: {total_smooth.min():.2f}\n"
        f"CE:    {ce_smooth.min():.2f}\n"
        f"KD:    {kd_smooth.min():.2f}"
    )

    plt.text(
        0.02,
        0.02,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
        fontfamily="monospace",
        color="#333333",
    )

    # Styling
    plt.title("Indra V11: The Quantum Pit (CE + KD)", fontsize=16, color="#333333")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        framealpha=0.9,
        fontsize=9,
        labelcolor="black",
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend


# Run Animation
ani = FuncAnimation(plt.gcf(), animate, interval=5000)  # Update every 5s
plt.show()
