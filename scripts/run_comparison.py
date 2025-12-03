import os
import subprocess
import pickle
import matplotlib.pyplot as plt
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    rc = process.poll()
    return rc

def main():
    print("=== Starting Comparison Workflow ===")
    
    # 1. Prepare Data
    print("\n--- Step 1: Data Preparation ---")
    if not os.path.exists("data/wikitext/train.pkl"):
        run_command(f"{sys.executable} scripts/prepare_data.py")
    else:
        print("Data already exists. Skipping.")
        
    # 2. Train Quantum Model
    print("\n--- Step 2: Training Quantum Model ---")
    run_command(f"{sys.executable} scripts/train_compare.py --model quantum --epochs 1")
    
    # 3. Train Baseline Model
    print("\n--- Step 3: Training Baseline Model ---")
    run_command(f"{sys.executable} scripts/train_compare.py --model baseline --epochs 1")
    
    # 4. Compare Results
    print("\n--- Step 4: Generating Comparison Plot ---")
    try:
        with open("logs/history_quantum.pkl", "rb") as f:
            hist_q = pickle.load(f)
        with open("logs/history_baseline.pkl", "rb") as f:
            hist_b = pickle.load(f)
            
        plt.figure(figsize=(10, 6))
        plt.plot(hist_q['loss'], label='IndraQuantum', marker='o')
        plt.plot(hist_b['loss'], label='Standard Baseline', marker='x')
        plt.title("Training Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        if not os.path.exists("plots"): os.makedirs("plots")
        plt.savefig("plots/training_comparison.png")
        print("Saved comparison plot to plots/training_comparison.png")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        
    print("\n=== Comparison Complete ===")

if __name__ == "__main__":
    main()
