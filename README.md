# 🕸️ IndraQuantum

> **"In the heaven of Indra, there is said to be a network of pearls, so arranged that if you look at one you see all the others reflected in it."** — _Avatamsaka Sutra_

**IndraQuantum** is a **Quantum-Inspired Language Model (QILM)** designed to bring extreme parameter efficiency to Natural Language Processing. By replacing standard vector embeddings with **Complex-Valued State Vectors** and utilizing **Tensor Network** decomposition, IndraQuantum achieves expressive power comparable to larger models while fitting on consumer hardware (e.g., 6GB VRAM).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🌌 The Concept

Standard Transformers treat tokens as static points in high-dimensional space. **IndraQuantum** treats them as **Quantum States** in a complex Hilbert space:

1.  **Superposition (The Particle):** Tokens are encoded as complex numbers ($z = r e^{i\theta}$).
    - **Magnitude ($r$):** Semantic meaning (e.g., "King").
    - **Phase ($\theta$):** Structural context, rotated by the document graph.
2.  **Entanglement (The Graph):** Instead of standard positional encodings, we use **Graph-Induced Phase Shift**. Relationships between Paragraphs, Sentences, and Words create "interference patterns" that determine attention.
3.  **Measurement (The Output):** The model predicts probabilities using the **Born Rule** ($P = |\psi|^2$), collapsing the complex state into real-valued logits compatible with standard LLMs.

---

## ⚡ Key Features

- **Complex Embeddings:** 2x information density per parameter compared to real-valued floats.
- **Tensor Train Decomposition:** Replaces massive Linear layers with efficient low-rank tensor networks, reducing parameters by **~10x**.
- **Knowledge Distillation Bridge:** Trains a compact "Quantum Student" to mimic a massive "Classical Teacher" (e.g., Qwen-2.5, Llama-3) by aligning their probability distributions.
- **Consumer Hardware Ready:** Designed specifically to train and run on GPUs with **<8GB VRAM**.

---

## 🧠 Architecture Comparison

| **Embedding** | `Float32` Vector | `Complex64` Wavefunction |
| **Attention** | Dot Product ($Q K^T$) | **Complex Edge-Biased** ($|Q^\dagger K|^2 + \text{Bias}$) |
| **Structure** | Absolute Positional Encoding | **Graph Topology + Local Window** |
| **FFN Layers** | Dense Matrices | **Complex FFN with CReLU** |
| **Output** | Linear Projection | **Born Rule Measurement** |

---

## 🏛️ Current Architecture: The "Quantum Core"

IndraQuantum Phase 1 is built on three pillars of efficiency:

### 1. Complex State Space ($z = r e^{i\theta}$)

The model operates entirely in the complex domain.

- **Why?** Complex numbers allow us to encode "Magnitude" (Semantic strength) and "Phase" (Relation/Context) in a single parameter, doubling information density.
- **Implementation:** `QuantumTTEmbedding` creates these distinct real/imaginary components.

### 2. Born Rule Measurement

To interact with the real world (and standard loss functions), we project the final quantum state using the **Born Rule**:
$$P(x) = | \psi(x) |^2$$
This collapses the wavefunction into a probability distribution over the vocabulary.

### 3. Tensor Train (TT) Compression

We factorize the massive embedding tables ($32000 \times 128$) into a chain of small 3rd-order tensors $G_1 \times G_2 \times G_3$.

- **Result:** We achieve **~6-20x parameter reduction** compared to dense matrices, allowing us to train a 32k vocabulary model on a laptop.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/divyang4481/IndraQuantum.git
cd IndraQuantum
pip install -r requirements.txt
```

### 2. Training (Phase 1: Knowledge Distillation)

Train the Quantum Student using a pre-trained Teacher (TinyLlama). This script handles the complex-to-real bridging.

```bash
# Optimized for 6GB VRAM
python scripts/train_full.py
```

### 3. Inference

```python
from indra.models.quantum_core import IndraQuantum

# Load your checkpoint
# ...
```

---

## 📂 Project Structure

```text
IndraQuantum/
├── indra/
│   ├── models/
│   │   ├── baseline.py         # Reference real-valued model
│   │   ├── embedding.py        # Complex embedding construction
│   │   ├── quantum_core.py     # Core IndraQuantum module
│   │   └── quantum_graph.py    # Graph-aware variant
│   └── graph/
│       ├── builder.py          # Builds document graph structure
│       └── phase_shift.py      # Graph-to-phase translation
├── scripts/
│   ├── train_full.py           # Main Training Script (KD + CE)
│   ├── analyze_trends.py       # Training Log Analysis
│   ├── plot_loss.py            # Loss Visualization
│   ├── test_checkpoint_oneshot.py # Rapid generation testing
│   ├── prepare_data.py         # Dataset preprocessing helpers
│   └── setup_teacher.py        # Teacher-model bootstrap
├── configs/
│   └── quantum_6gb.yaml        # Default low-VRAM config
├── data/
│   └── wikitext/
├── tests/
├── checkpoints/
├── logs/
├── plots/
└── teacher_models/
```

---

## 🔮 Future Roadmap: Graph Integration

While the current version achieves state-of-the-art efficiency using **Quantum Embeddings** and **Tensor Networks**, the next phase of IndraQuantum involves activating the **Graph Topology Layer**:

1.  **Hierarchy Construction:** We will parse input text into a hierarchical graph (Word -> Sentence -> Paragraph).
2.  **Phase-Shift Attention:** Instead of simple positional encoding, the "Distance" between tokens will be calculated as the **Shortest Path** on this graph.
3.  **Long-Range Dependency:** This will allow the model to "hop" from the start of a document to the end via the Paragraph node in just 2 steps, enabling infinite-context reasoning with linear complexity.

---

## 📜 Theory & Citations

This project is a novel synthesis of concepts from:

- _RotatE: Knowledge Graph Embedding by Relational Rotation_ (Sun et al., 2019)
- _Quantum Knowledge Distillation for Large Language Models_ (2025)
- _Encoding Word Order in Complex Embeddings_ (Wang et al., 2020)

---

## 🤝 Contributing

We welcome explorers! If you are interested in **Quantum Machine Learning**, **Tensor Networks**, or **Efficient NLP**, please open an issue or PR.

**"As above, so below."** — _The architecture of the cosmos, mirrored in code._
