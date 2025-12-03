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
    print("=== Starting Full Comparison Workflow (Baseline vs Quantum vs Graph) ===")
    
    # 1. Prepare Data
    print("\n--- Step 1: Data Preparation ---")
    if not os.path.exists("data/wikitext/train.pkl"):
        run_command(f"{sys.executable} scripts/prepare_data.py")
    else:
        print("Data already exists. Skipping.")
        
    # 2. Train Quantum Model
    print("\n--- Step 2: Training Quantum Model ---")
    if not os.path.exists("logs/history_quantum.pkl"):
        run_command(f"{sys.executable} scripts/train_compare.py --model quantum --epochs 1")
    else:
        print("Quantum history exists. Skipping.")
    
    # 3. Train Baseline Model
    print("\n--- Step 3: Training Baseline Model ---")
    if not os.path.exists("logs/history_baseline.pkl"):
        run_command(f"{sys.executable} scripts/train_compare.py --model baseline --epochs 1")
    else:
        print("Baseline history exists. Skipping.")
        
    # 4. Train Quantum Graph Model
    print("\n--- Step 4: Training Quantum Graph Model ---")
    if not os.path.exists("logs/history_graph.pkl"):
        run_command(f"{sys.executable} scripts/train_graph.py --epochs 1")
    else:
        print("Graph history exists. Skipping.")
    
    # 5. Compare Results
    print("\n--- Step 5: Generating Comparison Plot ---")
    try:
        data = {}
        if os.path.exists("logs/history_quantum.pkl"):
            with open("logs/history_quantum.pkl", "rb") as f: data['IndraQuantum'] = pickle.load(f)['loss']
        if os.path.exists("logs/history_baseline.pkl"):
            with open("logs/history_baseline.pkl", "rb") as f: data['Baseline'] = pickle.load(f)['loss']
        if os.path.exists("logs/history_graph.pkl"):
            with open("logs/history_graph.pkl", "rb") as f: data['IndraQuantum-Graph'] = pickle.load(f)['loss']
            
        plt.figure(figsize=(10, 6))
        for name, losses in data.items():
            plt.plot(losses, label=name, marker='o')
            
        plt.title("Training Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        if not os.path.exists("plots"): os.makedirs("plots")
        plt.savefig("plots/full_comparison.png")
        print("Saved comparison plot to plots/full_comparison.png")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== Comparison Complete ===")

if __name__ == "__main__":
    main()
