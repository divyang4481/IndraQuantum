import pandas as pd
import matplotlib.pyplot as plt

# Load the data
try:
    df = pd.read_csv("logs/training_metrics_rank64_optimized.csv")

    # Handle column naming differences
    if "KD_Logits" in df.columns:
        df["KD_Loss"] = df["KD_Logits"]

    # Group by Epoch to get the mean loss per epoch
    epoch_stats = df.groupby("Epoch").mean()

    print("\n=== Epoch-wise Average Metrics ===")
    cols_to_Show = ["Total_Loss", "CE_Loss", "KD_Loss"]
    print(epoch_stats[cols_to_Show].tail(15))

    # Calculate rate of change for the last 5 epochs
    last_5 = epoch_stats.tail(5)
    if len(last_5) > 1:
        kd_change = last_5["KD_Loss"].iloc[-1] - last_5["KD_Loss"].iloc[0]
        ce_change = last_5["CE_Loss"].iloc[-1] - last_5["CE_Loss"].iloc[0]

        print(f"\nChange in KD Loss (last 5 epochs): {kd_change:.4f}")
        print(f"Change in CE Loss (last 5 epochs): {ce_change:.4f}")
    else:
        print("\nNot enough data for trend analysis.")

except Exception as e:
    print(f"Error analyzing CSV: {e}")
