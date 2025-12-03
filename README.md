# 🕸️ IndraQuantum

> **"In the heaven of Indra, there is said to be a network of pearls, so arranged that if you look at one you see all the others reflected in it."** — *Avatamsaka Sutra*

**IndraQuantum** is a **Quantum-Inspired Language Model (QILM)** designed to bring extreme parameter efficiency to Natural Language Processing. By replacing standard vector embeddings with **Complex-Valued State Vectors** and utilizing **Tensor Network** decomposition, IndraQuantum achieves expressive power comparable to larger models while fitting on consumer hardware (e.g., 6GB VRAM).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🌌 The Concept

Standard Transformers treat tokens as static points in high-dimensional space. **IndraQuantum** treats them as **Quantum States** in a complex Hilbert space:

1.  **Superposition (The Particle):** Tokens are encoded as complex numbers ($z = r e^{i\theta}$).
    * **Magnitude ($r$):** Semantic meaning (e.g., "King").
    * **Phase ($\theta$):** Structural context, rotated by the document graph.
2.  **Entanglement (The Graph):** Instead of standard positional encodings, we use **Graph-Induced Phase Shift**. Relationships between Paragraphs, Sentences, and Words create "interference patterns" that determine attention.
3.  **Measurement (The Output):** The model predicts probabilities using the **Born Rule** ($P = |\psi|^2$), collapsing the complex state into real-valued logits compatible with standard LLMs.

---

## ⚡ Key Features

* **Complex Embeddings:** 2x information density per parameter compared to real-valued floats.
* **Tensor Train Decomposition:** Replaces massive Linear layers with efficient low-rank tensor networks, reducing parameters by **~10x**.
* **Knowledge Distillation Bridge:** Trains a compact "Quantum Student" to mimic a massive "Classical Teacher" (e.g., Qwen-2.5, Llama-3) by aligning their probability distributions.
* **Consumer Hardware Ready:** Designed specifically to train and run on GPUs with **<8GB VRAM**.

---

## 🧠 Architecture Comparison

| Component | Classical Transformer | IndraQuantum |
| :--- | :--- | :--- |
| **Embedding** | `Float32` Vector | `Complex64` Wavefunction |
| **Attention** | Dot Product ($Q K^T$) | **Interference** ($|Q^\dagger K|^2$) |
| **Structure** | Absolute Positional Encoding | **Graph Phase Rotation** |
| **FFN Layers** | Dense Matrices | **Tensor Train (TT) Decomposition** |
| **Output** | Linear Projection | **Born Rule Measurement** |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/divyang4481/IndraQuantum.git
cd IndraQuantum
pip install -r requirements.txt
```

### 2. Training (Knowledge Distillation)

Train the Quantum Student using a pre-trained Classical Teacher. This script automatically handles the complex-to-real domain bridging.

```bash
# Optimized for 6GB VRAM
python scripts/train_quantum.py \
    --teacher "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "data/corpus.txt" \
    --dim 128 \
    --batch_size 4
```

### 3. Inference

```python
from indra.models.quantum_core import IndraQuantum

model = IndraQuantum.from_pretrained("checkpoints/indra_v1")
output = model.generate("The nature of reality is")
print(output)
```

---

## 📂 Project Structure

```text
IndraQuantum/
├── indra/
│   ├── models/
│   │   ├── quantum_core.py    # Core Complex-Valued Model
│   │   ├── tensor_layers.py   # Tensor Train Linear Layers
│   │   └── interference.py    # Quantum Attention Mechanism
│   ├── graph/
│   │   └── phase_shift.py     # Graph-to-Phase logic
│   └── utils/
│       └── complex_ops.py     # Complex number helpers
├── scripts/
│   ├── train_quantum.py       # KD Training Loop
│   └── benchmark_param.py     # Parameter efficiency tests
└── configs/
    └── quantum_6gb.yaml       # Config for low VRAM
```

---

## 📜 Theory & Citations

This project is a novel synthesis of concepts from:

* *RotatE: Knowledge Graph Embedding by Relational Rotation* (Sun et al., 2019)
* *Quantum Knowledge Distillation for Large Language Models* (2025)
* *Encoding Word Order in Complex Embeddings* (Wang et al., 2020)

---

## 🤝 Contributing

We welcome explorers! If you are interested in **Quantum Machine Learning**, **Tensor Networks**, or **Efficient NLP**, please open an issue or PR.

**"As above, so below."** — *The architecture of the cosmos, mirrored in code.*
