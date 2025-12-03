import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title and Intro
    text_intro = """\
# IndraQuantum: Architecture Analysis & Comparison

This notebook analyzes the performance and characteristics of the **IndraQuantum** model compared to a standard baseline.
It also explores the **Quantum Graph** structure where text is modeled as a hierarchy of quantum systems (Tokens -> Sentences -> Paragraphs).

## 1. Initialization Analysis
We compare the "Quantum Polar Initialization" (Magnitude ~ 1, Random Phase) vs Standard Gaussian Initialization.
"""
    
    code_init = """\
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from indra.models.embedding import QuantumEmbedding

def analyze_initialization(vocab_size=1000, d_model=64):
    print("Analyzing Initialization...")
    
    # Quantum Embedding
    q_emb = QuantumEmbedding(vocab_size, d_model)
    q_weights = q_emb.embedding.weight.detach()
    real, imag = torch.chunk(q_weights, 2, dim=-1)
    q_mags = torch.sqrt(real**2 + imag**2).flatten().numpy()
    q_phases = torch.atan2(imag, real).flatten().numpy()
    
    # Standard Embedding
    s_emb = nn.Embedding(vocab_size, d_model * 2)
    s_weights = s_emb.weight.detach()
    s_real, s_imag = torch.chunk(s_weights, 2, dim=-1)
    s_mags = torch.sqrt(s_real**2 + s_imag**2).flatten().numpy()
    s_phases = torch.atan2(s_imag, s_real).flatten().numpy()
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].hist(q_mags, bins=50, alpha=0.7, label='Quantum (Polar Init)', color='blue')
    axs[0].hist(s_mags, bins=50, alpha=0.7, label='Standard (Normal Init)', color='orange')
    axs[0].set_title("Initialization Magnitude Distribution")
    axs[0].set_xlabel("Magnitude")
    axs[0].legend()
    
    axs[1].hist(q_phases, bins=50, alpha=0.7, label='Quantum', color='blue')
    axs[1].hist(s_phases, bins=50, alpha=0.7, label='Standard', color='orange')
    axs[1].set_title("Initialization Phase Distribution")
    axs[1].set_xlabel("Phase (Radians)")
    axs[1].legend()
    
    plt.show()

analyze_initialization()
"""

    # Training Comparison
    text_training = """\
## 2. Training Performance Comparison
We compare the training loss curves of:
1. **IndraQuantum**: Complex-valued MLP with Quantum Embeddings.
2. **Standard Baseline**: Real-valued MLP with same parameter count.
3. **IndraQuantum Graph**: Graph-based model with Token-Sentence-Paragraph hierarchy.

*Note: Models are trained using Knowledge Distillation from a TinyLlama teacher.*
"""

    code_training = """\
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

def plot_training_comparison():
    log_dir = "../logs"
    data = {}
    
    # Load Data
    files = {
        'IndraQuantum': 'history_quantum.pkl',
        'Baseline': 'history_baseline.pkl',
        'IndraQuantum-Graph': 'history_graph.pkl'
    }
    
    for name, filename in files.items():
        path = os.path.join(log_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                history = pickle.load(f)
                # Handle different formats if any (some might be dict, some list)
                if isinstance(history, dict):
                    data[name] = history['loss']
                else:
                    data[name] = history
        else:
            print(f"Warning: {filename} not found in {log_dir}")

    if not data:
        print("No training data found. Please run the training scripts first.")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    for name, losses in data.items():
        plt.plot(losses, label=name, marker='o', linewidth=2)
        
    plt.title("Training Loss Comparison (Knowledge Distillation)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (KL Divergence)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save plot to plots folder as well
    if not os.path.exists("../plots"):
        os.makedirs("../plots")
    plt.savefig("../plots/notebook_comparison_plot.png")
    plt.show()

plot_training_comparison()
"""

    # Graph Visualization
    text_graph = """\
## 3. Quantum Graph Structure
Visualizing the adjacency matrix of the text graph.
- **Nodes**: Tokens (0), Sentences (1), Paragraphs (2)
- **Edges**: Hierarchical connections (Token <-> Sentence <-> Paragraph)
"""

    code_graph = """\
from indra.graph.builder import TextGraphBuilder
from transformers import AutoTokenizer
import seaborn as sns

def visualize_graph_structure():
    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except:
        print("Could not load TinyLlama tokenizer, using dummy.")
        return

    builder = TextGraphBuilder(tokenizer)
    text = "Quantum mechanics is fascinating. It explains the universe.\\nThis is a new paragraph. It adds more context."
    
    graph = builder.build_graph(text)
    adj = graph['graph_mask'].numpy()
    node_types = graph['node_types'].numpy()
    
    print(f"Sequence Length: {len(node_types)}")
    print(f"Node Types: {node_types}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj, cmap="viridis", cbar=False)
    plt.title("Adjacency Matrix (Graph Mask)")
    plt.xlabel("Target Node Index")
    plt.ylabel("Source Node Index")
    plt.show()

visualize_graph_structure()
"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text_intro),
        nbf.v4.new_code_cell(code_init),
        nbf.v4.new_markdown_cell(text_training),
        nbf.v4.new_code_cell(code_training),
        nbf.v4.new_markdown_cell(text_graph),
        nbf.v4.new_code_cell(code_graph)
    ]
    
    if not os.path.exists("notebooks"):
        os.makedirs("notebooks")
        
    with open('notebooks/Architecture_Comparison.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook created at notebooks/Architecture_Comparison.ipynb")

if __name__ == "__main__":
    create_notebook()
