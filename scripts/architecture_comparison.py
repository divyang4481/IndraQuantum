import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indra.models.embedding import QuantumEmbedding

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def analyze_initialization(vocab_size=1000, d_model=64, save_dir="plots"):
    print("Analyzing Initialization...")
    
    # Quantum Embedding
    # d_model is the "complex" dimension, so output is d_model * 2
    q_emb = QuantumEmbedding(vocab_size, d_model)
    q_weights = q_emb.embedding.weight.detach() # [vocab, d_model * 2]
    
    # Calculate magnitudes (treating pairs as complex numbers)
    real, imag = torch.chunk(q_weights, 2, dim=-1)
    q_mags = torch.sqrt(real**2 + imag**2).flatten().numpy()
    
    # Standard Embedding
    # To be fair, let's use the same total number of parameters: d_model * 2
    s_emb = nn.Embedding(vocab_size, d_model * 2)
    s_weights = s_emb.weight.detach()
    # For standard embedding, "magnitude" is just the absolute value of the scalar, 
    # or we can treat adjacent pairs as complex to compare "vector magnitude"
    s_real, s_imag = torch.chunk(s_weights, 2, dim=-1)
    s_mags = torch.sqrt(s_real**2 + s_imag**2).flatten().numpy()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(q_mags, bins=50, alpha=0.7, label='Quantum (Polar Init)', color='blue')
    plt.hist(s_mags, bins=50, alpha=0.7, label='Standard (Normal Init)', color='orange')
    plt.title("Initialization Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Phase distribution
    q_phases = torch.atan2(imag, real).flatten().numpy()
    s_phases = torch.atan2(s_imag, s_real).flatten().numpy()
    
    plt.hist(q_phases, bins=50, alpha=0.7, label='Quantum', color='blue')
    plt.hist(s_phases, bins=50, alpha=0.7, label='Standard', color='orange')
    plt.title("Initialization Phase Distribution")
    plt.xlabel("Phase (Radians)")
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "initialization_comparison.png")
    plt.savefig(save_path)
    print(f"Saved initialization plot to {save_path}")
    plt.close()

def run_micro_benchmark(vocab_size=100, d_model=32, steps=200, save_dir="plots"):
    print("Running Micro-Benchmark (Convergence Test)...")
    
    # Task: Regression. Map Input ID -> Target Vector (Random)
    # This tests how quickly the embedding can adapt to represent a specific target.
    
    batch_size = 16
    target_dim = 16
    
    # Data
    input_ids = torch.randint(0, vocab_size, (batch_size * 10,)) # Fixed dataset
    targets = torch.randn(batch_size * 10, target_dim)
    
    # Model A: Quantum
    class QuantumModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = QuantumEmbedding(vocab_size, d_model)
            # Projection: (d_model * 2) -> target_dim
            self.proj = nn.Linear(d_model * 2, target_dim)
            
        def forward(self, x):
            return self.proj(self.emb(x))
            
    # Model B: Standard
    class StandardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model * 2) # Same param count
            self.proj = nn.Linear(d_model * 2, target_dim)
            
        def forward(self, x):
            return self.proj(self.emb(x))
            
    model_q = QuantumModel()
    model_s = StandardModel()
    
    opt_q = optim.Adam(model_q.parameters(), lr=0.01)
    opt_s = optim.Adam(model_s.parameters(), lr=0.01)
    
    criterion = nn.MSELoss()
    
    losses_q = []
    losses_s = []
    
    for i in range(steps):
        # Sample batch
        idx = torch.randint(0, len(input_ids), (batch_size,))
        batch_in = input_ids[idx]
        batch_target = targets[idx]
        
        # Train Q
        opt_q.zero_grad()
        out_q = model_q(batch_in)
        loss_q = criterion(out_q, batch_target)
        loss_q.backward()
        opt_q.step()
        losses_q.append(loss_q.item())
        
        # Train S
        opt_s.zero_grad()
        out_s = model_s(batch_in)
        loss_s = criterion(out_s, batch_target)
        loss_s.backward()
        opt_s.step()
        losses_s.append(loss_s.item())
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(losses_q, label='Quantum Embedding', color='blue')
    plt.plot(losses_s, label='Standard Embedding', color='orange')
    plt.title("Micro-Benchmark: Convergence Speed (Regression Task)")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "benchmark_convergence.png")
    plt.savefig(save_path)
    print(f"Saved benchmark plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    ensure_dir("plots")
    analyze_initialization()
    run_micro_benchmark()
